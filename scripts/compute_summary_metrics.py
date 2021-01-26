from glob import glob
import pandas as pd
import os
from ml_tools.evaluation import multi_class_eval, auc_with_nan, log_loss_with_labels
from tqdm import tqdm


base_dir = "./evaluations/test_standardised"
all_pred_files = glob(base_dir + "/*/obs_preds.csv")

for cur_pred_file in tqdm(all_pred_files):

    loaded = pd.read_csv(cur_pred_file, index_col=0)
    base_dir = os.path.split(cur_pred_file)[0]
    gt = pd.read_csv(base_dir + "/y_t.csv", index_col=0)
    auc_results = multi_class_eval(loaded, gt, auc_with_nan, "auc")
    log_loss_results = multi_class_eval(loaded, gt, log_loss_with_labels, "log_loss")
    combined = pd.concat([auc_results, log_loss_results], axis=1)

    target_file = base_dir + "/summary_metrics.csv"
    combined.to_csv(target_file)
