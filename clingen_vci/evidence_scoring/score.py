import re, argparse, sys, os
import numpy as np
import pandas as pd
import ast
from sklearn.metrics import precision_score, recall_score

sys.path.append("/home/users/oqueen/clingen_benchmark")

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data/"

from cgbench.cspec_version_utils import get_criteria_per_row
from cgbench.scoring_utils import precision_recall_at_k

def extract_evidence_code(text):
    # Remove text between think tags
    filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    match = re.search(r".*Evidence code:\s*([A-Z]+\d*(?:_[A-Za-z]+)?)", filtered_text)
    return match.group(1) if match else None

# def extract_evidence_code(text):
#     # Remove text between think tags
#     filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
#     match = re.search(r".*Evidence code:\s*([A-Z]+\d*)", filtered_text)
#     return match.group(1) if match else None

def main(args):
    rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_score/test_merged.csv"))
    
    # Handle multiple paths if provided
    if args.paths:
        dfs = []
        for path in args.paths:
            df = pd.read_csv(path)
            dfs.append(df)
        df_original = pd.concat(dfs, ignore_index=True)
    else:
        df_original = pd.read_csv(args.path)

    print("NUMBER OF SAMPLES: ", len(df_original))
    

    if args.one_pass:
        print("USING ONE PASS EVALUATION APPROACH - ONLY RECOMMENDED WITH REASONING MODELS")
        df = df_original.copy()
    else:
        print("USING MULTI-PASS EVALUATION")
        # Convert model_answer from string representation of list to actual list
        df_original["model_answer"] = df_original["model_answer"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Pass at k logic here
        # Explode the list to create separate rows for each element
        df = df_original.explode("model_answer")
        
        # Reset index after exploding
        df = df.reset_index(drop=True)

    # Break into intermediate codes
    for i in range(1, 4):
        df["model_answer_code_{}".format(i)] = df["model_answer"].apply(lambda x: extract_evidence_code(x)[:i] if extract_evidence_code(x) else None)

    df["model_answer_code_full"] = df["model_answer"].apply(lambda x: extract_evidence_code(x))

    # Get answers for all elements in a given row_id:
    # We know the df is repeated k times:
    #df_subset = df.drop_duplicates(subset=["evidence_code"], keep="first")
    gt_code_map = {row["full_row_id"]: ast.literal_eval(row["evidence_code"]) for _, row in df.iterrows()}

    unique_row_ids = df["full_row_id"].unique().tolist()

    results_dict = {
        #"full_row_id": [],
        "level": [],
        "precision": [],
        "recall": [],
    }

    latex_str = ""

    for level in range(1, 5):
        all_precision_level = []
        all_recall_level = []
        for row_id in unique_row_ids:
            all_row_answers = df.loc[df["full_row_id"] == row_id, :]
            gt_codes = gt_code_map[row_id] # This will be a list
            if level < 4:
                gt_codes = [gtc[:level] for gtc in gt_codes]
            
            # Load codes for this row_id:
            # Contained operation on evidence codes:
            evidence_codes = get_criteria_per_row(rel_df.iloc[row_id], DATA_PATH, stack_descriptions = False)
            if level < 4:
                evidence_codes[f"agg_code_{level}"] = evidence_codes["aggregate_code"].apply(lambda x: x[:level])
            else:
                evidence_codes[f"agg_code_{level}"] = evidence_codes["aggregate_code"]
            ecode_gt_unique = evidence_codes["agg_code_{}".format(level)].unique().tolist()
            ecode_tokenizer = {ecode: i for i, ecode in enumerate(ecode_gt_unique)}

            # Subset evidence codes to the level:
            if level < 4:
                model_answer_code = all_row_answers["model_answer_code_{}".format(level)]
            else:
                model_answer_code = all_row_answers["model_answer_code_full"]

            # Now perform retrieval-style scoring, i.e., retrieval and precision @k:
            # Make numpy arrays:
            gt_array = np.zeros(len(ecode_tokenizer))
            #import ipdb; ipdb.set_trace()
            for gtc in gt_codes:
                gt_array[ecode_tokenizer[gtc]] = 1
            
            # Now get the model answer code:
            model_answer_code_array = np.zeros((model_answer_code.shape[0], len(ecode_tokenizer)))
            for i, mac in enumerate(model_answer_code):
                try:
                    model_answer_code_array[i, ecode_tokenizer[mac]] = 1
                except KeyError:
                    # Don't assign it
                    continue
            
            precision_k, recall_k = precision_recall_at_k(gt_array, model_answer_code_array)
            # print(f"Level = {level}")
            # print(f"Precision @ k: {precision_k}")
            # print(f"Recall @ k: {recall_k}")
            all_precision_level.append(precision_k)
            all_recall_level.append(recall_k)

            # if gt_array.sum() > 1:
            #     import ipdb; ipdb.set_trace()

            #results_dict["full_row_id"].append(all_row_answers["full_row_id"].iloc[0])
            results_dict["level"].append(level)
            results_dict["precision"].append(precision_k)
            results_dict["recall"].append(recall_k)

        # Print median results with error
        median_precision = np.mean(all_precision_level)
        median_recall = np.mean(all_recall_level)
        precision_err = np.std(all_precision_level) / np.sqrt(len(all_precision_level))
        recall_err = np.std(all_recall_level) / np.sqrt(len(all_recall_level))
        
        print(f"\nMedian Results for Level {level}:")
        print(f"Precision: {median_precision:.3f} ± {precision_err:.3f}")
        print(f"Recall: {median_recall:.3f} ± {recall_err:.3f}\n")
            #import ipdb; ipdb.set_trace()

        precision_str = r"{" + f"{median_precision:.3f}" + r"}\std{" + f"{precision_err:.3f}" + r"}"
        recall_str = r"{" + f"{median_recall:.3f}" + r"}\std{" + f"{recall_err:.3f}" + r"}"

        #latex_str += f"& \{{median_precision:.3f}\}\std{precision_err:.3f} & \{{median_recall:.3f}\}\std{recall_err:.3f}\} & "
        if level < 4:
            latex_str = latex_str + f"{precision_str} & {recall_str} & "

    if args.save_path:
        rd = pd.DataFrame(results_dict)
        unique_levels = rd.level.unique().tolist()
        for level in unique_levels:
            for col in [f"precision_level{level}", f"recall_level{level}"]:
                #import ipdb; ipdb.set_trace()
                df_original[col] = rd[col.split("_")[0]].loc[rd["level"] == level]
        df_original.to_csv(args.save_path, index=False)
    
    print("LATEX:")
    print(latex_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Single path to process")
    parser.add_argument("--paths", nargs="+", type=str, help="Multiple paths to process and merge")
    parser.add_argument("--one_pass", action="store_true")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")

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