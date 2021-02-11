# set -e

# TO RUN OVERNIGHT

### VI
# python scripts/evaluate_ebird.py vi float gpu 1000 0 100 ./evaluations/comparisons/1000/
# python scripts/evaluate_ebird.py vi float gpu 2000 0 100 ./evaluations/comparisons/2000/
# python scripts/evaluate_ebird.py vi float gpu 4000 0 100 ./evaluations/comparisons/4000/
# python scripts/evaluate_ebird.py vi float gpu 8000 0 25 ./evaluations/comparisons/8000/
# python scripts/evaluate_ebird.py vi float gpu 16000 0 25 ./evaluations/comparisons/16000/

# This one didn't run. Need to reduce M.
python scripts/evaluate_ebird.py vi float gpu 32000 0 15 ./evaluations/comparisons/32000/

# python scripts/evaluate_ebird.py vi float gpu 64000 0 15 ./evaluations/comparisons/64000/
# python scripts/evaluate_ebird.py vi float gpu 100000 0 10 ./evaluations/comparisons/100000/

### Numpyro
# python scripts/evaluate_ebird.py numpyro float gpu 1000 0 100 ./evaluations/comparisons/1000/
# python scripts/evaluate_ebird.py numpyro float gpu 2000 0 100 ./evaluations/comparisons/2000/
# python scripts/evaluate_ebird.py numpyro float gpu 4000 0 100 ./evaluations/comparisons/4000/
python scripts/evaluate_ebird.py numpyro float gpu 8000 0 100 ./evaluations/comparisons/8000/
# python scripts/evaluate_ebird.py numpyro float gpu 16000 0 100 ./evaluations/comparisons/16000/
# python scripts/evaluate_ebird.py numpyro float gpu 32000 0 100 ./evaluations/comparisons/32000/

### Max lik
# python scripts/evaluate_ebird.py max_lik float gpu -1 0 100 ./evaluations/comparisons/max_lik_full/

### Stan
# python scripts/evaluate_ebird.py stan float gpu 1000 0 10 ./evaluations/comparisons/1000/

# Run some comparisons
# Floating point precision:
# python scripts/evaluate_ebird.py float gpu 180000 0 ./evaluations/comparisons_new_covs_daytime/float_gpu_180000_0_trust/
# python scripts/evaluate_ebird.py float gpu 2000 0 ./evaluations/comparisons/float_gpu_2000_0/
# python scripts/evaluate_ebird.py float gpu 4000 5 ./evaluations/comparisons/float_gpu_4000_5/
# python scripts/evaluate_ebird.py float gpu 8000 5 ./evaluations/comparisons/float_gpu_8000_5/
# 
# # Double precision
# python scripts/evaluate_ebird.py double gpu 1000 5 ./evaluations/comparisons/double_gpu_1000_5/
# python scripts/evaluate_ebird.py double gpu 2000 5 ./evaluations/comparisons/double_gpu_2000_5/
# python scripts/evaluate_ebird.py double gpu 4000 5 ./evaluations/comparisons/double_gpu_4000_5/
# python scripts/evaluate_ebird.py double gpu 8000 5 ./evaluations/comparisons/double_gpu_8000_5/
# 
# # CPU
# python scripts/evaluate_ebird.py float cpu 1000 5 ./evaluations/comparisons/float_cpu_1000_5/
# python scripts/evaluate_ebird.py float cpu 2000 5 ./evaluations/comparisons/float_cpu_2000_5/
# python scripts/evaluate_ebird.py float cpu 4000 5 ./evaluations/comparisons/float_cpu_4000_5/
# 
# # Big ones to finish
# python scripts/evaluate_ebird.py float gpu 16000 5 ./evaluations/comparisons/float_gpu_16000_5/
# python scripts/evaluate_ebird.py float gpu 32000 5 ./evaluations/comparisons/float_gpu_32000_5/

# python scripts/evaluate_ebird.py double gpu 16000 5 ./evaluations/comparisons/double_gpu_16000_5/
# python scripts/evaluate_ebird.py double gpu 32000 5 ./evaluations/comparisons/double_gpu_32000_5/
# python scripts/evaluate_ebird.py double gpu 64000 5 ./evaluations/comparisons/double_gpu_64000_5/

#python scripts/evaluate_ebird.py float gpu 40000 0 ./evaluations/comparisons_standardised/float_gpu_40000_0/
