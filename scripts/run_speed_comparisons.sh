set -e

# Run some comparisons
# Floating point precision:
python scripts/evaluate_ebird.py float gpu 1000 5 ./evaluations/comparisons/float_gpu_1000_5/
python scripts/evaluate_ebird.py float gpu 2000 5 ./evaluations/comparisons/float_gpu_2000_5/
python scripts/evaluate_ebird.py float gpu 4000 5 ./evaluations/comparisons/float_gpu_4000_5/
python scripts/evaluate_ebird.py float gpu 8000 5 ./evaluations/comparisons/float_gpu_8000_5/

# Double precision
python scripts/evaluate_ebird.py double gpu 1000 5 ./evaluations/comparisons/double_gpu_1000_5/
python scripts/evaluate_ebird.py double gpu 2000 5 ./evaluations/comparisons/double_gpu_2000_5/
python scripts/evaluate_ebird.py double gpu 4000 5 ./evaluations/comparisons/double_gpu_4000_5/
python scripts/evaluate_ebird.py double gpu 8000 5 ./evaluations/comparisons/double_gpu_8000_5/

# CPU
python scripts/evaluate_ebird.py float cpu 1000 5 ./evaluations/comparisons/float_cpu_1000_5/
python scripts/evaluate_ebird.py float cpu 2000 5 ./evaluations/comparisons/float_cpu_2000_5/
python scripts/evaluate_ebird.py float cpu 4000 5 ./evaluations/comparisons/float_cpu_4000_5/

# Big ones to finish
python scripts/evaluate_ebird.py float gpu 16000 5 ./evaluations/comparisons/float_gpu_16000_5/
python scripts/evaluate_ebird.py float gpu 32000 5 ./evaluations/comparisons/float_gpu_32000_5/

python scripts/evaluate_ebird.py double gpu 16000 5 ./evaluations/comparisons/double_gpu_16000_5/
python scripts/evaluate_ebird.py double gpu 32000 5 ./evaluations/comparisons/double_gpu_32000_5/
python scripts/evaluate_ebird.py double gpu 64000 5 ./evaluations/comparisons/double_gpu_64000_5/
