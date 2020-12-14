N_SPECIES=32
MIN_COUNT=5
OUTPUT_DIR=./investigations/stan_vs_variational_fixed_env_prior_more_species/

# for subset_size in 50 200 1000 2000; do
for subset_size in 100 1000; do

    python scripts/compare_stan_and_variational.py $subset_size $N_SPECIES $MIN_COUNT $OUTPUT_DIR

done
