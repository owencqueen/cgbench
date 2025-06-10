import pandas as pd
import numpy as np
import os, sys
import argparse
from dataclasses import dataclass

sys.path.append("/home/users/oqueen/clingen_benchmark")

from cgbench.explanation_judge import LLMExplanationJudge
from cgbench.prompts.gci_experimental_evidence import SOP_SCORE_RANGE
from cgbench.prompts.judge.gci import EVIDENCE_EXTRACTION_TASK_INPUT
from cgbench.gci_utils import parse_evidence_text, sop_to_lists

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data"
EVIDENCE_PATH = os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/evidence_cleaned_fulltext.csv")
ORIGINAL_EE_PATH = os.path.join(DATA_PATH, "GCI/Clingen-Gene-Disease-Summary-2025-03-31.csv")
original_ee_df = pd.read_csv(ORIGINAL_EE_PATH)

PUBMED_DF_PATH = os.path.join(DATA_PATH, "GCI/pubmed/experimental_evidence.csv")

SOP_MATCHED_LOOKUP = {k.replace("Model System Non-human model organism", "Model Systems Non-human model organism").lower(): v for k, v in SOP_SCORE_RANGE.items()}

@dataclass
class EvidenceExtraction:
    experimental_category: str
    explanation: str
    score: float
    score_adjustment_reason: str

#sop_path_gen = lambda sop_num: os.path.join(DATA_PATH, "GCI/SOP/experimental_evidence/{}.json".format(sop_num))

def build_evidence_prompt(row_in_ee, template = EVIDENCE_EXTRACTION_TASK_INPUT):

    evidence_prompt = template.format(
        gene = row_in_ee["GENE SYMBOL"],
        disease = row_in_ee["DISEASE LABEL"],
        mode_of_inheritance = row_in_ee["MOI"],
    )

    return evidence_prompt

    # TODO: Pick up here...

def build_paper_prompt(row, pm_df):
    pmid = row["pmid"]
    abstract = pm_df.loc[pmid, "abstract"]
    full_text = pm_df.loc[pmid, "full_text"]

    return EVIDENCE_PROMPT_PAPER.format(pmid=pmid, abstract=abstract, full_text=full_text)



def score_judge_explanation(
        args,
        model_answer_list, 
        gt_df, 
        match_matrix,
        judge_model, 
    ):
    
    # Find matches to iterate:
    matches_over_mal = match_matrix.any(axis=1)
    indices = np.arange(match_matrix.shape[0])[matches_over_mal]

    relative_mae_list = []
    directional_change_list = []

    #import ipdb; ipdb.set_trace()

    for i in indices:
        ma_i = model_answer_list[i]

        match_over_gt = match_matrix[i,:].astype(bool)
        if match_over_gt.sum() > 1:
            continue
        #import ipdb; ipdb.set_trace()
        match_gt_index = np.arange(match_matrix.shape[1])[match_over_gt][0].item()

        #import ipdb; ipdb.set_trace()

        # Score the "score" variable:
        ma_explanation = ma_i["explanation"]
        #import ipdb; ipdb.set_trace()
        gt_explanation = gt_df.iloc[match_gt_index]["Explanation"]
        row_in_ee = original_ee_df.loc[gt_df.iloc[match_gt_index]["primary_index"]]

        #gt_score, default_score = float(gt_score_full[0]), float(gt_score_full[1].replace("(", "").replace(")", ""))

        # Create input based on judge level
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

    return relative_mae_list, directional_change_list

def score_score(
        model_answer_list, 
        gt_df, 
        match_matrix,
        #ee_df
    ):
    
    # Find matches to iterate:
    matches_over_mal = match_matrix.any(axis=1)
    indices = np.arange(match_matrix.shape[0])[matches_over_mal]

    relative_mae_list = []
    directional_change_list = []

    #import ipdb; ipdb.set_trace()

    for i in indices:
        ma_i = model_answer_list[i]

        match_over_gt = match_matrix[i,:].astype(bool)
        if match_over_gt.sum() > 1:
            continue
        #import ipdb; ipdb.set_trace()
        match_gt_index = np.arange(match_matrix.shape[1])[match_over_gt][0].item()

        #import ipdb; ipdb.set_trace()

        # Score the "score" variable:
        ma_score = ma_i["score"]
        #import ipdb; ipdb.set_trace()
        gt_score_full = gt_df.iloc[match_gt_index]["Points (default points)"].split(" ")
        gt_score, default_score = float(gt_score_full[0]), float(gt_score_full[1].replace("(", "").replace(")", ""))

        try:
            _, sop_lb, sop_ub = SOP_MATCHED_LOOKUP[ma_i["category"].lower()]
        except KeyError:
            import ipdb; ipdb.set_trace()

        diff_score = ma_score - gt_score
        relative_mae = np.abs(diff_score) / (sop_ub - sop_lb)#/ gt_score
        relative_mae_list.append(relative_mae)

        diff_gt_default = gt_score - default_score
        diff_score_default = ma_score - default_score
        # Check if signs match between model-GT diff and default-GT diff
        if (diff_score_default > 0 and diff_gt_default > 0) or (diff_score_default < 0 and diff_gt_default < 0) or (diff_score_default == 0 and diff_gt_default == 0):
            directional_change_list.append(1)
        else:
            directional_change_list.append(0)

    return relative_mae_list, directional_change_list

    # Iterate over matches:

def determine_match(model_answer_list, gt_df, match_strategy = "category"):

    assert match_strategy in ["category", "judge_mediated_ec_filter", "judge_mediated_open"]
    match_matrix = np.zeros((len(model_answer_list), gt_df.shape[0]), dtype=float)

    total_llm_cost = 0.0

    match_metadata_dict = {
        "human_explanation": [],
        "human_explanation_id": [],
        "llm_explanation": [],
        "llm_explanation_id": [],
        "judge_pred": [],
        "judge_reasoning_concat": [],
    }

    #import ipdb; ipdb.set_trace()

    for i, ma_i in enumerate(model_answer_list):

        if match_strategy == "category":
            for j in range(gt_df.shape[0]):
                match_matrix[i, j] = float(gt_df.iloc[j]["Experimental Category"].lower().strip() == ma_i["category"].lower().strip())
        
        elif "judge_mediated" in match_strategy:

            # Instantiate judge:
            judge = LLMExplanationJudge(
                task = "gci_evidence_extraction",
                model_name = "gpt-4o-mini",
                temperature = 0.2,
                frequency_penalty = 0.0,
                best_of_k = 3,
            )

            if match_strategy == "judge_mediated_ec_filter":
                # First check if categories match
                for j in range(gt_df.shape[0]):
                    if gt_df.iloc[j]["Experimental Category"].lower().strip() == ma_i.experimental_category.lower().strip():
                        judge_pred, judge_out_list, llm_cost = judge(
                            llm_explanation = ma_i.explanation,
                            human_explanation = gt_df.iloc[j]["Explanation"],
                        )
                        match_matrix[i, j] = float(judge_pred)
                        total_llm_cost += llm_cost

                        match_metadata_dict["human_explanation"].append(gt_df.iloc[j]["Explanation"])
                        match_metadata_dict["human_explanation_id"].append(j)
                        match_metadata_dict["llm_explanation"].append(ma_i.explanation)
                        match_metadata_dict["llm_explanation_id"].append(i)
                        match_metadata_dict["judge_pred"].append(judge_pred)
                        match_metadata_dict["judge_reasoning_concat"].append("\n\n".join(judge_out_list))

                    else:
                        # Append non-matches:
                        match_metadata_dict["human_explanation"].append(gt_df.iloc[j]["Explanation"])
                        match_metadata_dict["human_explanation_id"].append(j)
                        match_metadata_dict["llm_explanation"].append(ma_i.explanation)
                        match_metadata_dict["llm_explanation_id"].append(i)
                        match_metadata_dict["judge_pred"].append(False)
                        match_metadata_dict["judge_reasoning_concat"].append(None)


            if match_strategy == "judge_mediated_open":
                for j in range(gt_df.shape[0]):
                    # Score no matter if the explanations match or not

                    judge_pred, judge_out_list, llm_cost = judge(
                        llm_explanation = ma_i.explanation,
                        human_explanation = gt_df.iloc[j]["Explanation"],
                    )

                    match_matrix[i, j] = float(judge_pred)
                    total_llm_cost += llm_cost

                    match_metadata_dict["human_explanation"].append(gt_df.iloc[j]["Explanation"])
                    match_metadata_dict["human_explanation_id"].append(j)
                    match_metadata_dict["llm_explanation"].append(ma_i.explanation)
                    match_metadata_dict["llm_explanation_id"].append(i)
                    match_metadata_dict["judge_pred"].append(judge_pred)
                    match_metadata_dict["judge_reasoning_concat"].append("\n\n".join(judge_out_list))

    if "judge_mediated" in match_strategy:
        match_metadata_df = pd.DataFrame(match_metadata_dict)
    else:
        match_metadata_df = None

    return match_matrix, total_llm_cost, match_metadata_df

def main(args):

    # Handle multiple paths if provided
    if args.paths:
        dfs = []
        for path in args.paths:
            df = pd.read_csv(path)
            dfs.append(df)
        results_df = pd.concat(dfs, ignore_index=True)
    else:
        results_df = pd.read_csv(args.path)

    #ee_df = pd.read_csv(ORIGINAL_EE_PATH)
    #results_df["category"] = results_df["model_answer"].apply(lambda x: [parse_evidence_text(xi)["category"] for xi in x])

    print("Number of samples: {}".format(results_df.shape[0]))

    gb_results_df = results_df.groupby(["primary_index", "pmid"])

    gt_df = pd.read_csv(EVIDENCE_PATH)
    gb_gt_df = gt_df.groupby(["primary_index", "pmid"])
    gt_df_map = {key: df_i for key, df_i in gb_gt_df}

    running_recall_list, running_precision_list = [], []

    total_cost_sum = 0.0
    relative_mae_list_allsamples = []
    directional_change_list_allsamples = []
    pd_metadata_list = []
    over_count = 0
    extraction_failure_count = 0

    for (primary_index, pmid), result_df in gb_results_df:
        #model_answer_list = [parse_evidence_text(row["model_answer"]) for _, row in result_df.iterrows()]
        model_answer_list = parse_evidence_text(result_df["model_answer"].iloc[0])
        #model_answer_list = [x["category"].replace("Model System Non-human model organism", "Model Systems Non-human model organism") for x in model_answer_list]
        if model_answer_list == -1:
            extraction_failure_count += 1
            if not args.ignore_missing_extraction:
                running_recall_list.append(0.0)
                running_precision_list.append(0.0)
            continue

        for i, x in enumerate(model_answer_list):
            model_answer_list[i]["category"] = x["category"].replace("Model System Non-human model organism", "Model Systems Non-human model organism")


        #map_to = (row["primary_index"], row["pmid"])
        gt_df_i = gt_df_map[(primary_index, pmid)]

        matches, total_llm_cost, match_metadata_df = determine_match(model_answer_list, gt_df_i, match_strategy = args.match_strategy)

        if args.run_judge:
            

        #import ipdb; ipdb.set_trace()
        over = (matches.sum(axis=1) > 1).any()
        if over:
            over_count += 1

        relative_mae_list, directional_change_list = score_score(model_answer_list, gt_df_i, matches)
        relative_mae_list_allsamples.extend(relative_mae_list)
        directional_change_list_allsamples.extend(directional_change_list)

        if match_metadata_df is not None:
            match_metadata_df["primary_index"] = primary_index
            match_metadata_df["pmid"] = pmid

            pd_metadata_list.append(match_metadata_df)

        recall = matches.any(axis=0).sum() / matches.shape[1]
        precision = matches.any(axis=1).sum() / matches.shape[0]

        running_recall_list.append(recall)
        running_precision_list.append(precision)

        #print(f"Recall: {recall}, Precision: {precision}")
        #print("Cost:", total_llm_cost)

        total_cost_sum += total_llm_cost

    print("AVERAGE RESULTS")
    print("Recall:                     {:.3f} ± {:.3f}".format(np.mean(running_recall_list), np.std(running_recall_list)/np.sqrt(len(running_recall_list))))
    print("Precision:                  {:.3f} ± {:.3f}".format(np.mean(running_precision_list), np.std(running_precision_list)/np.sqrt(len(running_precision_list))))
    print("Success extractions:        {} ({:.2f}%)".format((results_df.shape[0] - extraction_failure_count), (results_df.shape[0] - extraction_failure_count)/results_df.shape[0] * 100))
    print("\n")
    print("Number of extractions scored: {}".format(len(relative_mae_list_allsamples)))
    print("Score MAE:                  {:.3f} ± {:.3f}".format(np.mean(relative_mae_list_allsamples), np.std(relative_mae_list_allsamples)/np.sqrt(len(relative_mae_list_allsamples))))
    print("Matched directional change: {:.3f} ± {:.3f}".format(np.mean(directional_change_list_allsamples), np.std(directional_change_list_allsamples)/np.sqrt(len(directional_change_list_allsamples))))
    print("Cost:", total_cost_sum)

    print("Over count", over_count)

    if ("judge_mediated" in args.match_strategy) and (args.output_path is not None):
        pd_metadata_df = pd.concat(pd_metadata_list)
        pd_metadata_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Single path to process")
    parser.add_argument("--paths", nargs="+", type=str, help="Multiple paths to process and merge")
    parser.add_argument("--match_strategy", type=str, required=False, default = "category", choices = ["category", "judge_mediated_ec_filter", "judge_mediated_open"])
    parser.add_argument("--output_path", type=str, required=False, default = None)

    parser.add_argument("--ignore_missing_extraction", action="store_true")

    parser.add_argument("--run_judge", action="store_true")
    parser.add_argument("--judge_model_name", type=str, required=False, default="gpt-4o-mini")
    parser.add_argument("--judge_temperature", type=float, required=False, default=0.5)
    parser.add_argument("--judge_frequency_penalty", type=float, required=False, default=0.0)
    parser.add_argument("--judge_best_of_k", type=int, required=False, default=1)
    parser.add_argument("--judge_awareness_level", type=str, required=False, default="general", choices=["general", "task_aware", "evidence_aware"])

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
