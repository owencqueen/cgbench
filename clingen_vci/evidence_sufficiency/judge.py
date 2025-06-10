import re, argparse, sys, os
import numpy as np
import pandas as pd

sys.path.append("/home/users/oqueen/clingen_benchmark")
sys.path.append("/home/users/oqueen/clingen_benchmark/clingen_vci/evidence_sufficiency")

from cgbench.prompts.judge import JUDGE_PROMPT_MAP_VCI_EVER, GENERAL_JUDGE_SYSTEM_PROMPT

from cgbench.vci_utils import build_vcep_map, find_row_in_vcep_map
from cgbench.explanation_judge import LLMExplanationJudge
from cgbench.prompts.vci_evidence_suff import EVIDENCE_PROMPT_WITH_LABEL, EVIDENCE_CODE_TEMPLATE

from cgbench.prompts.judge.vci_ever import EVIDENCE_PROMPT_TASK_AWARE, EVIDENCE_PROMPT_PAPER
from cgbench.cspec_version_utils import get_criteria_per_row

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data" # TODO: move to env
VCEP_DIRECTORY_PATH = os.path.join(DATA_PATH, "VCI/csr_criteria/cspec_directory_processed.csv")
VCEP_CSV_DIRPATH = os.path.join(DATA_PATH, "VCI/csr_criteria/vcep_csv")
PMID_TEXT_PATH = os.path.join(DATA_PATH, "VCI/pubmed_id_to_text.csv")

def extract_prediction(text):
    match = re.search(r'Prediction:\s*"?(not met|met)"?', text)
    return match.group(1).replace(" ", "") if match else None

MET_TOKENIZER = {
    "met": 1,
    "notmet": 0
}

def combo_extract_tokenize(text):
    try:
        #import ipdb; ipdb.set_trace()
        ecode = extract_prediction(text)
        return MET_TOKENIZER[ecode]
    except:
        print(ecode)
        return -1

def extract_explanation(text):
    match = re.search(r'Explanation:\s*(.*?)(?=(?:Prediction:|$))', text, re.DOTALL)
    if match:
        explanation = match.group(1).strip()
        # Remove any quotes around the explanation
        explanation = explanation.strip('"')
        return explanation
    return None

def build_evidence_prompt(row, template = EVIDENCE_PROMPT_TASK_AWARE):
    evidence_code = row["evidence_code"]
    pmid = row["pmid"]
    variant = row["variant"]
    disease = row["disease"]
    inheritance = row["mode_inheritance"]

    evidence_codes_df = get_criteria_per_row(row, DATA_PATH, stack_descriptions = False)
    evidence_code_desc = evidence_codes_df.loc[evidence_codes_df["aggregate_code"] == evidence_code.replace("-", "_"), "Description"].values[0]

   # import ipdb; ipdb.set_trace()

    evidence_prompt = template.format(
        variant=variant,
        disease=disease,
        inheritance=inheritance,
        evidence_code = EVIDENCE_CODE_TEMPLATE.format(ecode=evidence_code, desc=evidence_code_desc),
        prediction=row["met_status"].replace("_", " ")
    )

    return evidence_prompt

def build_paper_prompt(row, pm_df):
    pmid = row["pmid"]
    abstract = pm_df.loc[pmid, "abstract"]
    full_text = pm_df.loc[pmid, "full_text"]

    return EVIDENCE_PROMPT_PAPER.format(pmid=pmid, abstract=abstract, full_text=full_text)

def main(args):
    # Load and potentially merge multiple dataframes
    if args.paths:
        dfs = []
        for path in args.paths:
            df = pd.read_csv(path)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(args.path)

    # Handle deduplication if requested
    #if args.dedup_adjustment:
    filter_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/test_dedup.csv"))
    filter_df["id_tuple"] = filter_df.apply(lambda x: (x["entry_index"], x["evidence_code"]), axis=1)
    df["id_tuple"] = df.apply(lambda x: (x["entry_index"], x["evidence_code"]), axis=1)

    mask = df["id_tuple"].isin(filter_df["id_tuple"])
    df = df[mask]

    filter_in = filter_df["id_tuple"].isin(df["id_tuple"])
    filter_in_not_included = (~filter_in).sum()

    rel_df = filter_df[filter_in]#.set_index("id_tuple")

    print(f"Num samples (before filter): {df.shape[0]}")
    print(f"Num samples (after filter): {df.shape[0]}")
    print(f"Num samples (not included): {filter_in_not_included}")

    # Set MOI:
    df["mode_inheritance"] = [filter_df["mode_inheritance"][filter_df["id_tuple"] == row["id_tuple"]].values[0] for i, row in df.iterrows()]
    df["variant"] = [filter_df["variant"][filter_df["id_tuple"] == row["id_tuple"]].values[0] for i, row in df.iterrows()]
    df["path"] = [filter_df["path"][filter_df["id_tuple"] == row["id_tuple"]].values[0] for i, row in df.iterrows()]

        #import ipdb; ipdb.set_trace()

    # elif args.dedup_filter:
    #     print("Num samples (before filter):", df.shape[0])

    #     rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/test.csv"))
    #     df["entry_index"] = df["full_row_id"].apply(lambda i: rel_df.loc[i, "entry_index"])

    #     df_dedup = pd.read_csv(os.path.join(DATA_PATH, "VCI/clingen_vci_pubmed_fulltext_dedup_pmid.csv"))
    #     row_df = [(row["entry_index"], row["evidence_code"]) for _, row in df.iterrows()]
    #     row_df_dedup = [(row["entry_index"], row["evidence_code"]) for _, row in df_dedup.iterrows()]

    #     mask = df.apply(lambda row: (row["entry_index"], row["evidence_code"]) in row_df_dedup, axis=1)
    #     df = df[mask]

    # Load original df with explanation data:
    #rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/clingen_vci_pubmed_var_na_filtered.csv"))

    print("Num samples:", df.shape[0])

    # Process predictions and labels
    df["label"] = df["met_status"].apply(lambda x: MET_TOKENIZER[x.strip().replace(" ", "").replace("_", "")])
    df["prediction"] = df["model_answer"].apply(combo_extract_tokenize)

    print(f"NUM INVALID: {np.sum(df['prediction'] == -1)}")

    # Filter out invalid predictions if not including all explanations
    if not args.include_all_explanations:
        mask = df["prediction"] != -1
        df = df.loc[mask]
        df = df.loc[df["label"] == df["prediction"]]

    # Run judge over all explanations + their ground truths:
    judge = LLMExplanationJudge(
        system_prompt = GENERAL_JUDGE_SYSTEM_PROMPT,
        model_name = args.judge_model_name,
        temperature = args.judge_temperature,
        frequency_penalty = args.judge_frequency_penalty,
        double_input = True,
        best_of_k = args.judge_best_of_k,
    )

    pm_df = pd.read_csv(PMID_TEXT_PATH)
    pm_df.set_index("pmid", inplace=True)

    results_dict = {
        "judgement": [],
        "explanation": [],
    }

    #import ipdb; ipdb.set_trace()

    for i, row in df.iterrows():
        human_explanation = rel_df.loc[rel_df["id_tuple"] == row["id_tuple"], args.field_to_compare_against].values[0]
        #human_explanation = rel_df.loc[row["id_tuple"], args.field_to_compare_against]
        explanation = extract_explanation(row["model_answer"])

        if args.judge_awareness_level == "general":
            prefix_prompt = None
        else:
            task_prompt = build_evidence_prompt(row)

            if args.judge_awareness_level == "task_aware":
                prefix_prompt = JUDGE_PROMPT_MAP_VCI_EVER["task_aware"].format(
                    input_description_template = task_prompt,
                )
            elif args.judge_awareness_level == "evidence_aware":
                prefix_prompt = JUDGE_PROMPT_MAP_VCI_EVER["evidence_aware"].format(
                    input_description_template = task_prompt,
                    pubmed_info = build_paper_prompt(row, pm_df),
                )

            import ipdb; ipdb.set_trace()

        judgement, llm_out_list, llm_cost_total = judge(
            llm_explanation = explanation,
            human_explanation = human_explanation,
            prefix_prompt = prefix_prompt,
        )

        import ipdb; ipdb.set_trace()

        concat_llm_explanations = "\n\n".join(llm_out_list)
        
        results_dict["judgement"].append(judgement)
        results_dict["explanation"].append(concat_llm_explanations)

    results_df = pd.DataFrame(results_dict)
    results_df = pd.concat([df, results_df], axis=1, ignore_index=True)
    results_df.to_csv(args.save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--paths", nargs="+", type=str, help="Multiple paths to process and merge")
    parser.add_argument("--save_path", type=str, default="judge_out.csv")

    parser.add_argument("--include_all_explanations", action="store_true", help="Default behavior is to only measure on matched explanations")
    parser.add_argument("--field_to_compare_against", type=str, default="summary_comments", choices=["summary_comments", "comments", "summary"])

    parser.add_argument("--judge_model_name", type=str, required=False, default="gpt-4o-mini")
    parser.add_argument("--judge_temperature", type=float, required=False, default=0.5)
    parser.add_argument("--judge_frequency_penalty", type=float, required=False, default=0.0)
    parser.add_argument("--judge_best_of_k", type=int, required=False, default=1)

    parser.add_argument("--dedup_filter", action="store_true", help="Deduplicate the results")
    parser.add_argument("--dedup_adjustment", action="store_true", help="Filter to examples from a specific DF")

    parser.add_argument("--judge_awareness_level", type=str, required=False, default="general", choices=["general", "task_aware", "evidence_aware"])

    args = parser.parse_args()

    #assert args.dedup_adjustment, "TODO: Fix later"
    args.dedup_adjustment = True
    
    if not args.path and not args.paths:
        parser.error("Either --path or --paths must be provided")
    if args.path and args.paths:
        parser.error("Cannot use both --path and --paths simultaneously")

    if args.paths:
        print(f"Judging merged results from {len(args.paths)} paths")
    else:
        print(f"Judging results from {args.path}")

    main(args)