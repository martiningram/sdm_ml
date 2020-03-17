data_dir=./data
result_dir=./results

min_observations=5

mogp_n_inducing=20
mogp_n_latent=4
mogp_seed=2
mogp_test_run=1
mogp_target_dir=$(result_dir)/mogp_n_inducing_$(mogp_n_inducing)_n_latent_$(mogp_n_latent)_seed_$(mogp_seed)_is_test_run_$(mogp_test_run)

# To load modules required for R on spartan:
# module load R/3.5.2-spartan_gcc-6.2.0 GDAL UDUNITS cairo zlib Pango

$(mogp_target_dir)/results.npz: fit_mogp.py $(data_dir)/train_covariates.csv
	python fit_mogp.py \
		$(data_dir)/train_covariates.csv \
		$(data_dir)/train_outcomes.csv \
		$(mogp_target_dir) \
		$(mogp_n_inducing) \
		$(mogp_n_latent) \
		$(mogp_seed) \
		$(mogp_test_run)

$(data_dir)/train_covariates.csv: prepare_data.py is_test.csv
	python prepare_data.py \
		is_test.csv \
		$(data_dir) \
		$(min_observations)

is_test.csv: make_split.Rmd
	Rscript -e "library(knitr); knit('make_split.Rmd')"
