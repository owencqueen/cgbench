import re, argparse, sys, os, ast
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append("/home/users/oqueen/clingen_benchmark")
sys.path.append("/home/users/oqueen/clingen_benchmark/clingen_vci/evidence_sufficiency")
sys.path.append("/Users/owenqueen/Desktop/stanford_research/SHERLOCK_HOME/clingen_benchmark")

from cgbench.prompts.judge import JUDGE_PROMPT_MAP_VCI_ESCORE, GENERAL_JUDGE_SYSTEM_PROMPT

from cgbench.vci_utils import build_vcep_map, find_row_in_vcep_map
from cgbench.explanation_judge import LLMExplanationJudge
from cgbench.prompts.vci_evidence_suff import EVIDENCE_PROMPT_WITH_LABEL, EVIDENCE_CODE_TEMPLATE

from cgbench.prompts.judge.vci_escore import EVIDENCE_PROMPT_TASK_AWARE, EVIDENCE_PROMPT_PAPER
from cgbench.cspec_version_utils import get_criteria_per_row

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data" # TODO: move to en
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/Users/owenqueen/Desktop/stanford_research/SHERLOCK_HOME/clingen_benchmark/data"

VCEP_DIRECTORY_PATH = os.path.join(DATA_PATH, "VCI/csr_criteria/cspec_directory_processed.csv")
VCEP_CSV_DIRPATH = os.path.join(DATA_PATH, "VCI/csr_criteria/vcep_csv")
PMID_TEXT_PATH = os.path.join(DATA_PATH, "VCI/pubmed_id_to_text.csv")

def extract_evidence_code(text):
    # Remove text between think tags
    filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    match = re.search(r".*Evidence code:\s*([A-Z]+\d*(?:_[A-Za-z]+)?)", filtered_text)
    return match.group(1) if match else None

def choose_correct_rows(exp_df):
    exp_df["model_answer"] = exp_df["model_answer"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    exp_df["evidence_code"] = exp_df["evidence_code"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    exp_df = exp_df.explode("model_answer")
    exp_df = exp_df.explode("evidence_code")

    #import ipdb; ipdb.set_trace()

    exp_df["model_code_answer"] = exp_df["model_answer"].apply(lambda x: extract_evidence_code(x))

    correct_rows = []
    counter = 0
    for i in exp_df["full_row_id"].unique():
        df_subset = exp_df[exp_df["full_row_id"] == i]
        # Expand corresponding:


        corresponding = (df_subset["model_code_answer"] == df_subset["evidence_code"])

        #import ipdb; ipdb.set_trace()

        if corresponding.sum() == 0:
            counter += 1
            continue

        #assert corresponding.sum() > 0, "There should be at least one correct answer"

        #try:
        correct_rows.append(df_subset[corresponding])
        # except:
        #     import ipdb; ipdb.set_trace()

    print(f"Number of rows with no correct answer: {counter}")

    # Concat all:
    correct_df = pd.concat(correct_rows)
    #import ipdb; ipdb.set_trace()
    return correct_df

    #return pd.DataFrame(correct_rows)

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
    evidence_code = row["model_code_answer"]
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

    # Filter out invalid predictions if not including all explanations
    #if not args.include_all_explanations:
    df = choose_correct_rows(df)

    if args.reverse_df:
        df = df.iloc[::-1]

    #import ipdb; ipdb.set_trace()
    #df["evidence_code"] = df["evidence_code"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_score/test_merged.csv"))

    df["full_entry_index"] = [rel_df.loc[i, "entry_index"] for i in df["full_row_id"]]

    rel_df[args.field_to_compare_against] = rel_df[args.field_to_compare_against].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    rel_df["evidence_code"] = rel_df["evidence_code"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Zip the lists together and explode to create paired rows
    # e.g. if evidence_code=['PS1','PM2'] and field_to_compare=['exp1','exp2']
    # it creates 2 rows: (PS1,exp1), (PM2,exp2)
    rel_df['zipped'] = rel_df.apply(lambda x: list(zip(x['evidence_code'], x[args.field_to_compare_against])), axis=1)
    rdf_exp = rel_df.explode('zipped')
    rdf_exp[['evidence_code', args.field_to_compare_against]] = pd.DataFrame(rdf_exp['zipped'].tolist(), index=rdf_exp.index)
    rdf_exp = rdf_exp.drop('zipped', axis=1)

    rdf_by_pmid_entry_code = rdf_exp.set_index(["pmid", "entry_index", "evidence_code"]).sort_index()

    # for i, row in df.iterrows():
    #     # Choose out gt_explanation:
    #     gt_exp_list = rel_df.loc[row["full_row_id"], "gt_explanation"]
        

    df["gt_explanation"] = [rdf_by_pmid_entry_code.loc[(row["pmid"], row["full_entry_index"], row["evidence_code"]), args.field_to_compare_against] for i, row in df.iterrows()]
    #df["gt_explanation"] = df.apply(lambda row: row["gt_explanation"][row["evidence_code"].index(row["model_code_answer"])], axis=1)

    df["mode_inheritance"] = [rel_df.loc[i, "mode_inheritance"] for i in df["full_row_id"]]
    df["variant"] = [rel_df.loc[i, "variant"] for i in df["full_row_id"]]
    df["path"] = [rel_df.loc[i, "path"] for i in df["full_row_id"]]

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
    total_cost = 0
    last_intermediate_path = None

    #import ipdb; ipdb.set_trace()

    # Reset df index:
    df = df.reset_index(drop=True)

    COUNTER = 0
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        human_explanation = row["gt_explanation"] #rel_df.loc[row["full_row_id"], args.field_to_compare_against]
        explanation = extract_explanation(row["model_answer"])

        #import ipdb; ipdb.set_trace()

        if args.judge_awareness_level == "general":
            prefix_prompt = None
        else:
            task_prompt = build_evidence_prompt(row)

            if args.judge_awareness_level == "task_aware":
                prefix_prompt = JUDGE_PROMPT_MAP_VCI_ESCORE["task_aware"].format(
                    input_description_template = task_prompt,
                )
            elif args.judge_awareness_level == "evidence_aware":
                prefix_prompt = JUDGE_PROMPT_MAP_VCI_ESCORE["evidence_aware"].format(
                    input_description_template = task_prompt,
                    pubmed_info = build_paper_prompt(row, pm_df),
                )

        #import ipdb; ipdb.set_trace()

        judgement, llm_out_list, llm_cost_total = judge(
            llm_explanation = explanation,
            human_explanation = human_explanation,
            prefix_prompt = prefix_prompt,
        )

        total_cost += llm_cost_total

        #import ipdb; ipdb.set_trace()

        concat_llm_explanations = "\n<SEP>\n".join(llm_out_list)
        
        results_dict["judgement"].append(judgement)
        results_dict["explanation"].append(concat_llm_explanations)

        # Save intermediate results every 25 steps
        if (COUNTER + 1) % 25 == 0:
            # Delete previous intermediate file if it exists
            if last_intermediate_path and os.path.exists(last_intermediate_path):
                os.remove(last_intermediate_path)
                print(f"Deleted previous intermediate file: {last_intermediate_path}")

            intermediate_results_df = pd.DataFrame(results_dict)
            intermediate_results_df = pd.concat([df.iloc[:COUNTER+1], intermediate_results_df], axis=1)
            intermediate_save_path = args.save_path.replace('.csv', f'_intermediate_{COUNTER+1}.csv')
            intermediate_results_df.to_csv(intermediate_save_path, index=False)
            print(f"Saved intermediate results at step {COUNTER+1} to {intermediate_save_path}")
            print(f"Current correct: {intermediate_results_df.judgement.sum()}")
            print(f"Total cost: {total_cost:.5f}")
            
            # Update the last intermediate path
            last_intermediate_path = intermediate_save_path

        COUNTER += 1

    results_df = pd.DataFrame(results_dict)
    results_df = pd.concat([df, results_df], axis=1)
    
    # Delete the last intermediate file if it exists
    if last_intermediate_path and os.path.exists(last_intermediate_path):
        os.remove(last_intermediate_path)
        print(f"Deleted last intermediate file: {last_intermediate_path}")
    
    results_df.to_csv(args.save_path, index=False)
    print(f"Total cost: {total_cost:.5f}")

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

    parser.add_argument("--reverse_df", action="store_true", help="Reverse the order of the dataframe")

    parser.add_argument("--dedup_filter", action="store_true", help="Deduplicate the results")
    parser.add_argument("--dedup_adjustment", action="store_true", help="Filter to examples from a specific DF")

    parser.add_argument("--judge_awareness_level", type=str, required=False, default="general", choices=["general", "task_aware", "evidence_aware"])

    args = parser.parse_args()

    #assert args.dedup_adjustment, "TODO: Fix later"
    
    if not args.path and not args.paths:
        parser.error("Either --path or --paths must be provided")
    if args.path and args.paths:
        parser.error("Cannot use both --path and --paths simultaneously")

    if args.paths:
        print(f"Judging merged results from {len(args.paths)} paths")
    else:
        print(f"Judging results from {args.path}")

    main(args)