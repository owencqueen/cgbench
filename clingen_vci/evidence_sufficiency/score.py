import re, argparse, sys, os
import pandas as pd

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils import resample

sys.path.append("/home/users/oqueen/clingen_benchmark")
sys.path.append("/home/users/oqueen/clingen_benchmark/clingen_vci/evidence_sufficiency")

from prompts import *

from cgbench.vci_utils import build_vcep_map, find_row_in_vcep_map

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data" # TODO: move to env
PMID_TEXT_PATH = os.path.join(DATA_PATH, "VCI/pubmed_id_to_text.csv")

def extract_prediction(text):
    # Find the "Prediction:" tag and extract everything after it, accounting for surrounding punctuation
    filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    match = re.search(r'\*{0,2}Prediction\*{0,2}:', text)
    if not match:
        return None
    text_after_prediction = text[match.end():]
    
    # Search for met/not met in the text after "Prediction:"
    match = re.search(r'[\s\*]*"?(not met|met)"?', text_after_prediction, re.IGNORECASE)
    return match.group(1).strip().lower().replace(" ", "").replace("_", "") if match else None

MET_TOKENIZER = {
    "met": 1,
    "notmet": 0
}

def combo_extract_tokenize(text):
        #import ipdb; ipdb.set_trace()
    ecode = extract_prediction(text)
    if ecode is None:
        return -1
    try:
        return MET_TOKENIZER[ecode]
    except KeyError:
        print("\nERROR:")
        print(text)
        return -1

def bootstrap_metric(y_true, y_pred, metric_func, B=1000, random_state=None):
    rng = np.random.RandomState(random_state)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(B):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        vals.append(metric_func(y_true[sample_idx], y_pred[sample_idx]))
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1)

# Metric functions for TPR and TNR

def tpr_score(y_true, y_pred):
    # True Positive Rate: TP / (TP + FN)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TP = cm[1,1]
    FN = cm[1,0]
    return TP / (TP + FN) if (TP + FN) > 0 else np.nan

def tnr_score(y_true, y_pred):
    # True Negative Rate: TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN = cm[0,0]
    FP = cm[0,1]
    return TN / (TN + FP) if (TN + FP) > 0 else np.nan

def main(args):

    

    if args.paths:
        dfs = []
        for path in args.paths:
            df = pd.read_csv(path)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(args.path)

    if args.dedup_adjustment:
        filter_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/test_dedup.csv"))
        filter_df["id_tuple"] = filter_df.apply(lambda x: (x["entry_index"], x["evidence_code"]), axis=1)
        df["id_tuple"] = df.apply(lambda x: (x["entry_index"], x["evidence_code"]), axis=1)

        mask = df["id_tuple"].isin(filter_df["id_tuple"])
        df = df[mask]

        filter_in = filter_df["id_tuple"].isin(df["id_tuple"])

        filter_in_not_included = (~filter_in).sum()

        print(f"Num samples (before filter): {df.shape[0]}")
        print(f"Num samples (after filter): {df.shape[0]}")
        print(f"Num samples (not included): {filter_in_not_included}")

    elif args.dedup_filter:
        print("Num samples (before filter):", df.shape[0])

        rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/test.csv"))
        df["entry_index"] = df["full_row_id"].apply(lambda i: rel_df.loc[i, "entry_index"])

        # TEMPORARY: perform sampling down:
        df_dedup = pd.read_csv(os.path.join(DATA_PATH, "VCI/clingen_vci_pubmed_fulltext_dedup_pmid.csv"))
        #df_dedup = df_dedup[df_dedup["pmid"].isin(df["pmid"])]
        row_df = [(row["entry_index"], row["evidence_code"]) for _, row in df.iterrows()]
        row_df_dedup = [(row["entry_index"], row["evidence_code"]) for _, row in df_dedup.iterrows()]

        mask = df.apply(lambda row: (row["entry_index"], row["evidence_code"]) in row_df_dedup, axis=1)

        df = df[mask]
        #import ipdb; ipdb.set_trace()

    # Get a random sample of 1000 rows from df_dedup


    print("Num samples:", df.shape[0])

    #import ipdb; ipdb.set_trace()


    df["label"] = df["met_status"].apply(lambda x: MET_TOKENIZER[x.strip().lower().replace(" ", "").replace("_", "")])
    df["prediction"] = df["model_answer"].apply(combo_extract_tokenize)

    #import ipdb; ipdb.set_trace()

    # Calculate and print F1 score
    print(f"NUM INVALID: {np.sum(df['prediction'] == -1)}")

    mask = df["prediction"] == -1
    #df = df.loc[mask]

    # Set masked to wrong
    df.loc[mask, "prediction"] = 1 - df["label"][mask]

    # Calculate and print F1, TPR, and TNR with error bars
    f1, f1_se = bootstrap_metric(df["label"].to_numpy(), df["prediction"].to_numpy(),
                                 lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary'))
    tpr, tpr_se = bootstrap_metric(df["label"].to_numpy(), df["prediction"].to_numpy(), tpr_score)
    tnr, tnr_se = bootstrap_metric(df["label"].to_numpy(), df["prediction"].to_numpy(), tnr_score)

    print(f"F1 Score: {f1:.3f} ± {f1_se:.3f}")
    print(f"TPR (Recall/Sensitivity): {tpr:.3f} ± {tpr_se:.3f}")
    print(f"TNR (Specificity): {tnr:.3f} ± {tnr_se:.3f}")
    #print(f"Positive rate = {}")

    f1_str = r"{" + f"{f1:.3f}" + r"}" + r"\std{" + f"{f1_se:.3f}" + r"}"
    tnr_str = r"{" + f"{tnr:.3f}" + r"}" + r"\std{" + f"{tnr_se:.3f}" + r"}"
    tpr_str = r"{" + f"{tpr:.3f}" + r"}" + r"\std{" + f"{tpr_se:.3f}" + r"}"

    print(tnr_str + " & " + tpr_str + " & " + f1_str)

    print(f"P Rate = {df.prediction.mean()} +- {df.prediction.std() / np.sqrt(df.shape[0])}")

    # Print confusion matrix
    cm = confusion_matrix(df["label"], df["prediction"])
    
    print("\nConfusion Matrix:")
    print("-----------------")
    print("             Predicted")
    print("             Not Met  Met")
    print(f"Actual Not Met   {cm[0][0]:<7} {cm[0][1]}")
    print(f"      Met        {cm[1][0]:<7} {cm[1][1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--paths", nargs="+", type=str, help="Multiple paths to process and merge")

    parser.add_argument("--dedup_filter", action="store_true", help="Deduplicate the results")

    parser.add_argument("--dedup_adjustment", action="store_true", help="Filter to examples from a specific DF")

    args = parser.parse_args()
    
    if not args.path and not args.paths:
        parser.error("Either --path or --paths must be provided")
    if args.path and args.paths:
        parser.error("Cannot use both --path and --paths simultaneously")

    if args.paths:
        print(f"Scoring merged results from {len(args.paths)} paths")
    else:
        print(f"Score of {args.path}")

    main(args)