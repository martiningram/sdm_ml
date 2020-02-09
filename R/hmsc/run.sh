dataset_name="$1"
target_dir="$2"
n_transient="$3"
n_sample="$4"
n_thin="$5"
n_chains="$6"
add_poly_terms="$7"
add_interaction_terms="$8"

Rscript hmsc.R \
  "$dataset_name" \
  "$target_dir" \
  "$n_transient" \
  "$n_sample" \
  "$n_thin" \
  "$n_chains" \
  "$add_poly_terms" \
  "$add_interaction_terms"
