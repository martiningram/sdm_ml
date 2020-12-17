import pandas as pd

# Compare environment coefficients
unmarked_coefs = pd.read_csv("./state_coefs_unmarked.csv", index_col=0)
fast_coefs = pd.read_csv("./state_coefs_fast.csv", index_col=0)

print("Checking environment coefficients...")
diffs = pd.Series(
    unmarked_coefs.values[:, 0] - fast_coefs.values[:, 0], index=unmarked_coefs.index
)

assert all(diffs.values < 1e-3)
print("All differences are smaller than 1e-3")

# Compare detection coefficients
unmarked_coefs = pd.read_csv("./det_coefs_unmarked.csv", index_col=0)
fast_coefs = pd.read_csv("./det_coefs_fast.csv", index_col=0)

print("Checking detection coefficients...")
diffs = pd.Series(
    unmarked_coefs.values[:, 0] - fast_coefs.values[:, 0], index=unmarked_coefs.index
)

assert all(diffs.values < 1e-3)
print("All differences are smaller than 1e-3")
