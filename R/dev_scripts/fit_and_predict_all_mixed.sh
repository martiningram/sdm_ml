dataset_dir="./datasets"

for subdir in "$dataset_dir"/*; do

  echo "Fitting and predicting $subdir ..."

  cur_train_x="$subdir/x_train.csv"
  cur_train_y="$subdir/y_train.csv"
  cur_test_x="$subdir/x_test.csv"
  cur_target_dir="$subdir"

  Rscript fit_and_predict_mixed.R \
    --train-x-csv "$cur_train_x" \
    --train-y-csv "$cur_train_y" \
    --test-x-csv "$cur_test_x" \
    --target-dir "$cur_target_dir"

done
