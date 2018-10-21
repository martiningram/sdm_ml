import time
import numpy as np
import pickle
import os
import tensorflow as tf
import tensorflow_probability as tfp
from sdm_ml.model import PresenceAbsenceModel
from sdm_ml.gp.utils import find_starting_z
from sklearn.preprocessing import StandardScaler
from ml_tools.tf_kernels import ard_rbf_kernel_batch
from sdm_ml.gp.tf_utils import (compute_cholesky_if_possible,
                                cholesky_solve_if_possible)

tf.enable_eager_execution()


class DTCGaussianProcessHMC(PresenceAbsenceModel):

    def __init__(self, n_inducing, verbose=True, full_test_conditional=False,
                 jitter=0., n_chains=2, n_leapfrog_steps=8, n_warmup=500,
                 n_sampling=500):

        if full_test_conditional:
            raise Exception('Predicting with full test conditional not'
                            ' yet implemented!')

        self.scaler = None
        self.verbose = verbose
        self.n_inducing = n_inducing
        self.full_test_conditional = full_test_conditional
        self.jitter = jitter
        self.n_chains = n_chains
        self.n_leapfrog_steps = n_leapfrog_steps
        self.n_sampling = n_sampling
        self.n_warmup = n_warmup
        self.samples = None
        self.diagnostics = None
        self.is_fit = False

    def define_parameters(self, n_data, n_cov):

        self.f = tf.get_variable(
            'f', shape=(self.n_chains, self.n_inducing, 1),
            initializer=tf.random_normal_initializer(0., 0.1))

        self.lengthscales = tf.get_variable(
            'lengthscales', shape=(self.n_chains, n_cov,),
            initializer=tf.random_uniform_initializer(-1., 1.),
            dtype=tf.float32)

        self.alpha = tf.get_variable(
            'alpha', dtype=tf.float32, shape=(self.n_chains,),
            initializer=tf.random_uniform_initializer(-1., 1.))

        self.bias = tf.get_variable(
            'bias', dtype=tf.float32, shape=(self.n_chains,),
            initializer=tf.random_uniform_initializer(-1., 1.))

        self.f_step_size = tf.get_variable(
            'f_step', shape=self.f.get_shape(),
            initializer=tf.constant_initializer(1e-2))

        self.alpha_step_size = tf.get_variable(
            'alpha_step', shape=self.alpha.get_shape(),
            initializer=tf.constant_initializer(1e-2))

        self.lengthscale_step = tf.get_variable(
            'lengthscale_step', shape=self.lengthscales.get_shape(),
            initializer=tf.constant_initializer(1e-2))

        self.bias_step = tf.get_variable(
            'bias_step', shape=self.bias.get_shape(),
            initializer=tf.constant_initializer(1e-2))

    def fit(self, X, y):

        self.samples = []
        self.diagnostics = []
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        y = y.astype(np.float32)

        n_data, n_cov = X.shape

        self.inducing = find_starting_z(
            X, self.n_inducing, use_minibatching=False)
        self.inducing = tf.constant(self.inducing.astype(np.float32))

        X = tf.constant(X, dtype=tf.float32)

        for i in range(y.shape[1]):

            cur_y = tf.constant(y[:, [i]], dtype=tf.float32)

            # Prepare the model
            self.define_parameters(n_data, n_cov)

            def curried_posterior(u, alpha, lengthscales, bias):

                return self.log_posterior(u, alpha, lengthscales, bias,
                                          X, cur_y, self.inducing)

            update_policy = tfp.mcmc.make_simple_step_size_update_policy

            hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=curried_posterior,
                num_leapfrog_steps=self.n_leapfrog_steps,
                step_size=[self.f_step_size, self.alpha_step_size,
                           self.lengthscale_step, self.bias_step],
                step_size_update_fn=update_policy()
            )

            start_time = time.time()
            self.step = 0

            samples, _ = tfp.mcmc.sample_chain(
                num_results=self.n_sampling,
                num_burnin_steps=self.n_warmup,
                current_state=[self.f, self.alpha, self.lengthscales,
                               self.bias],
                kernel=hmc
            )

            r_hats = {x: tfp.mcmc.potential_scale_reduction(y) for x, y in
                      zip(['f', 'alpha', 'lengthscales', 'bias'], samples)}

            if self.verbose:
                print(r_hats)
                print('Run time was: {} seconds including r_hat.'.format(
                    time.time() - start_time))

            # Convert eager tensors to numpy
            samples = [x.numpy() for x in samples]

            (u_samples, alpha_samples, lengthscale_samples,
             bias_samples) = samples

            # Put them back on the constrained scale
            alpha_samples = np.exp(alpha_samples)
            lengthscale_samples = np.exp(lengthscale_samples)

            # Concatenate the samples to drop the chain dimensions
            self.samples.append({
                 'u_raw': np.concatenate(u_samples.transpose(1, 0, 2, 3)),
                 'alpha': np.concatenate(alpha_samples.T),
                 'lengthscales': np.concatenate(
                     lengthscale_samples.transpose(1, 0, 2)),
                 'bias': np.concatenate(bias_samples.T)
             })

            self.diagnostics.append(r_hats)

        self.is_fit = True

    def predict(self, X):

        assert(self.is_fit)

        X = tf.constant(self.scaler.transform(X), dtype=tf.float32)

        y_pred = list()

        for cur_samples in self.samples:

            cur_means = list()

            for cur_u_raw, cur_a, cur_l, cur_b in zip(
                    cur_samples['u_raw'], cur_samples['alpha'],
                    cur_samples['lengthscales'], cur_samples['bias']):

                cur_l = tf.expand_dims(cur_l, axis=0)
                cur_a = tf.expand_dims(cur_a, axis=0)

                cur_kuu = ard_rbf_kernel_batch(
                    self.inducing, self.inducing, cur_l, cur_a,
                    jitter=self.jitter)[0]

                cur_kuu_chol = tf.linalg.cholesky(cur_kuu)
                cur_kuu_inv = tf.linalg.cholesky_solve(
                    cur_kuu_chol, tf.eye(cur_kuu_chol.shape[0].value))

                cur_k_star_u = ard_rbf_kernel_batch(
                    X, self.inducing, cur_l, cur_a, jitter=self.jitter)[0]

                cur_u = tf.matmul(cur_kuu_chol, cur_u_raw)

                mean_pred = tf.matmul(
                    cur_k_star_u, tf.matmul(cur_kuu_inv, cur_u)) + cur_b

                cur_means.append(mean_pred.numpy())

            cur_means = np.array(cur_means)

            cur_predictions = np.mean(tf.sigmoid(cur_means), axis=0)

            y_pred.append(np.squeeze(cur_predictions))

        y_pred = np.stack(y_pred, axis=1)

        return np.squeeze(y_pred)

    def save_parameters(self, target_folder):

        to_pickle = {
            'samples': self.samples,
            'diagnostics': self.diagnostics
            }

        with open(os.path.join(target_folder, 'dtc_gp.pkl'), 'wb') as f:
            pickle.dump(to_pickle, f)

    def log_posterior(self, u, alpha, lengthscales, bias, x, y, inducing):

        if self.step % 100 == 0 and self.verbose:
            print('Iteration {}'.format(self.step))

        self.step += 1

        # JACOBIAN ADJUSTMENT
        # Inputs are assumed to be unconstrained. Constrain now.
        log_jac_adj = alpha + tf.reduce_sum(lengthscales, axis=1)

        alpha = tf.exp(alpha)
        lengthscales = tf.exp(lengthscales)

        # PRIORS
        alpha_prior = tfp.distributions.InverseGamma(3., 5.)
        lengthscale_prior = tfp.distributions.InverseGamma(3., 5.)
        bias_prior = tfp.distributions.Normal(0., 1.)
        u_prior = tf.distributions.Normal(0., 1.)

        log_prior = (
            alpha_prior.log_prob(alpha) +
            tf.reduce_sum(lengthscale_prior.log_prob(lengthscales), axis=1) +
            bias_prior.log_prob(bias) +
            tf.reduce_sum(u_prior.log_prob(u), axis=(1, 2)) +
            log_jac_adj
        )

        if self.verbose:
            alpha = tf.Print(alpha, [alpha, lengthscales, bias])

        k_ind = ard_rbf_kernel_batch(inducing, inducing, lengthscales, alpha,
                                     jitter=self.jitter)

        ind_chols = tf.map_fn(compute_cholesky_if_possible, k_ind)

        n_ind = tf.constant(ind_chols.get_shape()[1], dtype=tf.int32)
        n_batch = tf.constant([ind_chols.get_shape()[0]], dtype=tf.int32)

        rhs = tf.eye(n_ind, batch_shape=n_batch)

        cholesky_inverse = tf.map_fn(
            cholesky_solve_if_possible, [ind_chols, rhs], dtype=tf.float32)

        transposed_inverse = tf.transpose(cholesky_inverse, (0, 2, 1))

        k_fu = (ard_rbf_kernel_batch(x, inducing, lengthscales, alpha,
                                     jitter=self.jitter))

        L = tf.matmul(k_fu, transposed_inverse)

        # Samples for the inducing point values
        f_samples = tf.matmul(L, u)
        f_samples = f_samples + tf.reshape(bias, (-1, 1, 1))

        log_lik = y * f_samples - f_samples + tf.log_sigmoid(f_samples)

        log_lik = tf.squeeze(tf.reduce_sum(log_lik, axis=1))
        log_post = log_prior + log_lik

        return log_post
