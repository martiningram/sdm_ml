import numpy as np
import tensorflow as tf


def solve_via_cholesky_if_possible(arguments):

    x, y = arguments

    try:
        cholesky = tf.cholesky(x)
        solved = tf.cholesky_solve(cholesky, y)
        return solved
    except Exception:
        print('Linear algebra error!')
        return tf.transpose(tf.ones((y.get_shape()[1], x.get_shape()[0]))
                            * np.nan)


def compute_cholesky_if_possible(x):

    try:
        cholesky = tf.cholesky(x)
        return cholesky
    except Exception:
        print('Matrix not positive-definite!')
        return tf.ones_like(x) * np.nan


def cholesky_solve_if_possible(arguments):

    x, y = arguments

    try:
        solved = tf.cholesky_solve(x, y)
        return solved
    except Exception:
        print('Could not solve Cholesky.')
        return tf.ones((x.get_shape()[0], y.get_shape()[1])) * np.nan
