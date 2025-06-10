import os, sys, argparse, ast
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("/home/users/oqueen/clingen_benchmark")

from cgbench.util_agents import OnePassAgent
from cgbench.prompts.vci_evidence_suff import *
from cgbench.cspec_version_utils import get_criteria_per_row
from cgbench.best_of_k import BestofKAgent

#from cgbench.vci_utils import build_vcep_map, find_row_in_vcep_map

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data" # TODO: move to env
VCEP_DIRECTORY_PATH = os.path.join(DATA_PATH, "VCI/csr_criteria/cspec_directory_processed.csv")
VCEP_CSV_DIRPATH = os.path.join(DATA_PATH, "VCI/csr_criteria/vcep_csv")
PMID_TEXT_PATH = os.path.join(DATA_PATH, "VCI/pubmed_id_to_text.csv")
PMID_SUMMARIZED_TEXT_PATH = os.path.join(DATA_PATH, "VCI/small_llm_summarization/summarized_pubmed.csv")

# VCEP_DIRECTORY_PATH = "/home/users/oqueen/clingen_vci/data/csr_criteria/cspec_directory_processed.csv"
# VCEP_CSV_DIRPATH = "/home/users/oqueen/clingen_vci/data/csr_criteria/vcep_csv"
# PMID_TEXT_PATH = "/home/users/oqueen/clingen_vci/data/pubmed_id_to_text.csv"

def construct_evidence_code_str(evidence_codes):

    # Get instruction sys prompt:
    evidence_code_str = ""
    for _, rowi in evidence_codes.iterrows():
        ecode_i = EVIDENCE_CODE_TEMPLATE.format(ecode=rowi["aggregate_code"], desc=rowi["Description"])
        evidence_code_str += ecode_i
    return evidence_code_str

def construct_prompt_str(row, pm_sub, input_template, explanation = None, pm_summarized_df = None):
    evidence_code = row["evidence_code"]
    pmid = row["pmid"]
    variant = row["variant"]
    disease = row["disease"]
    inheritance = row["mode_inheritance"]
    abstract = pm_sub.loc[pmid, "abstract"]
    if pm_summarized_df is not None:
        key = row["unifying_row_mapper"]
        # Handle case where key matches multiple rows - take first summary
        if isinstance(pm_summarized_df.loc[key, "summary"], pd.Series):
            full_text = pm_summarized_df.loc[key, "summary"].iloc[0]
        else:
            full_text = pm_summarized_df.loc[key, "summary"]
        #full_text = pm_summarized_df.loc[key, "summary"].iloc[0] if isinstance(pm_summarized_df.loc[key, "summary"], pd.Series) else pm_summarized_df.loc[key, "summary"]
    else:
        full_text = pm_sub.loc[pmid, "full_text"]
    met_prediction = row["met_status"].replace("_", " ")


    if explanation is None:
        evidence_prompt = input_template.format(
            variant=variant,
            disease=disease,
            inheritance=inheritance,
            pmid=pmid,
            abstract=abstract,
            full_text=full_text,
            evidence_code=evidence_code,
            prediction = met_prediction,
        )
    else:
        evidence_prompt = input_template.format(
            variant=variant,
            disease=disease,
            inheritance=inheritance,
            pmid=pmid,
            abstract=abstract,
            full_text=full_text,
            evidence_code=evidence_code,
            prediction = met_prediction,
            explanation = explanation
        )

    return evidence_prompt

def build_icl_examples(
        args, 
        icl_explanation: bool = False,
    ):

    # Choose example template:
    if args.icl_omit_fulltext:
        example_template = EVIDENCE_PROMPT_WITH_LABEL_ABSTRACT_EXPLANATION if args.icl_explanation else EVIDENCE_PROMPT_WITH_LABEL_ABSTRACT
    elif args.icl_omit_paper:
        example_template = EVIDENCE_PROMPT_NOPM_WITH_LABEL_EXPLANATION if args.icl_explanation else EVIDENCE_PROMPT_NOPM_WITH_LABEL
    else:
        # Else include the paper
        example_template = EVIDENCE_PROMPT_WITH_LABEL_EXPLANATION if args.icl_explanation else EVIDENCE_PROMPT_WITH_LABEL

    if args.dedup_filter:
        train_examples = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/train_dedup.csv"))
    else:
        train_examples = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/train.csv"))

    train_examples["expert_panel"] = train_examples["expert_panel"].apply(lambda x: x.lower().replace(" ", "").replace("/", "") if x is not None else None)

    # train_examples["evidence_code"] = train_examples["evidence_code"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # train_examples["summary_comments"] = train_examples["summary_comments"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    #train_examples["comments"] = train_examples["comments"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    #train_examples["summary"] = train_examples["summary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Need to explode multiple rows - summary_comments, comments, summary, and evidence_code
    #train_examples = train_examples.explode(["summary_comments", "evidence_code"])

    pm_df = pd.read_csv(PMID_TEXT_PATH)

    pm_summarized_df = None
    if args.icl_summarize_paper:
        #pm_df = pd.read_csv(PMID_TEXT_PATH)
        pm_summarized_df = pd.read_csv(PMID_SUMMARIZED_TEXT_PATH)

        # Create a mapper based on gene, diseas, and mode of inheritance:
        pm_summarized_df["unifying_row_mapper"] = pm_summarized_df["pmid"] + "_" + pm_summarized_df["variant"] + "_" + pm_summarized_df["disease"] + "_" + pm_summarized_df["inheritance"]
        pm_summarized_df.set_index("unifying_row_mapper", inplace=True)
        # Create a mapping dictionary from pmid to summary
        # summary_map = dict(zip(pm_summarized_df['pmid'], pm_summarized_df['summary']))
        # # Map summaries to pm_df using the dictionary
        # pm_df["full_text"] = pm_df["pmid"].map(summary_map)

        # Temp: check that all rel_df samples are in here:
        train_examples["unifying_row_mapper"] = train_examples["pmid"] + "_" + train_examples["variant"] + "_" + train_examples["disease"] + "_" + train_examples["mode_inheritance"]
        assert train_examples["unifying_row_mapper"].isin(pm_summarized_df.index).all(), "All rel_df samples must be in pm_summarized_df"

    pm_sub = pm_df.loc[pm_df.pmid.isin(train_examples.pmid)]
    pm_sub.set_index("pmid", inplace=True) # Create our own subset of the pm lookup
    
    # Get unique evidence codes and their counts
    evidence_codes = train_examples['evidence_code'].unique()
    
    # Initialize random number generator
    rng = np.random.RandomState(args.icl_sample_seed)
    
    selected_indices = []
    
    # First select one example from each unique evidence code
    for code in evidence_codes:
        code_indices = train_examples[train_examples['evidence_code'] == code].index
        selected_indices.append(rng.choice(code_indices))
        
    # If we need more examples, distribute remaining slots evenly
    remaining = args.icl_shot_number - len(evidence_codes)
    if remaining > 0:
        # Calculate how many additional examples per code
        per_code = remaining // len(evidence_codes)
        leftover = remaining % len(evidence_codes)
        
        for code in evidence_codes:
            code_indices = train_examples[train_examples['evidence_code'] == code].index
            code_indices = code_indices[~np.isin(code_indices, selected_indices)]
            
            if len(code_indices) > 0:
                # Add per_code examples for this evidence code
                n_select = per_code + (1 if leftover > 0 else 0)
                n_select = min(n_select, len(code_indices))
                selected_indices.extend(rng.choice(code_indices, size=n_select, replace=False))
                leftover -= 1

    #import ipdb; ipdb.set_trace()
    train_subset = train_examples.iloc[selected_indices[:args.icl_shot_number],:]

    # Build out full strings:
    icl_string = ""

    counter = 1
    for i, rowi in train_subset.iterrows():

        #evidence_codes = find_row_in_vcep_map(rowi, vcep_map).sort_values("label")
        evidence_codes = get_criteria_per_row(rowi, DATA_PATH, stack_descriptions = True)

        #evidence_code_str = construct_evidence_code_str(evidence_codes)

        row_evidence_code = evidence_codes.loc[evidence_codes["aggregate_code"] == rowi["evidence_code"].replace("-", "_").replace(" ", ""),:]

        evidence_code_str = EVIDENCE_CODE_TEMPLATE.format(ecode=row_evidence_code["aggregate_code"].values[0], desc=row_evidence_code["Description"].values[0])

        # Evidence code str is just the appropriate one:
        # evidence_code_gt = rowi["evidence_code"].replace("-", "_").replace(" ", "")
        # #try:
        # gt_mask = (evidence_codes["aggregate_code"] == evidence_code_gt)
        # if gt_mask.sum() == 0:
        #     print("ERROR: SKIPPING BECAUSE NO CODE IS FOUND")
        #     continue
        # #try:
        # evidence_code_desc = evidence_codes.loc[gt_mask, "Description"].values[0]

        if icl_explanation:
            explanation = rowi["summary_comments"] # FIX
        else:
            explanation = None

        evidence_prompt = construct_prompt_str(rowi, pm_sub, example_template, explanation, pm_summarized_df)
        if args.icl_code_description:
            icl_string += f"Example {counter}:\nRelevant evidence code for Example {counter}:\n{evidence_code_str}\n{evidence_prompt}\n\n"
        else:
            icl_string += f"Example {counter}:\n{evidence_prompt}\n\n"
        counter += 1

    return icl_string

def save_intermediate_results(results_df, save_path, step_num, previous_file=None):
    """Save intermediate results with step number in the filename and delete the previous intermediate file."""
    base_path, ext = os.path.splitext(save_path)
    intermediate_path = f"{base_path}_step={step_num}{ext}"
    
    # Delete the previous intermediate file if it exists
    if previous_file and os.path.exists(previous_file):
        try:
            os.remove(previous_file)
            print(f"Deleted previous intermediate file: {previous_file}")
        except Exception as e:
            print(f"Warning: Could not delete previous intermediate file {previous_file}: {e}")
    
    # Save the new intermediate file
    results_df.to_csv(intermediate_path, index=False)
    print(f"Saved intermediate results to {intermediate_path}")
    
    return intermediate_path

def main(args):

    if args.dedup_filter:
        rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/test_dedup.csv"))
    else:
        rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/split_evidence_sufficiency/test.csv"))
    #rel_df = pd.read_csv(os.path.join(DATA_PATH, "VCI/clingen_vci_pubmed_fulltext.csv"))
    rel_df["expert_panel"] = rel_df["expert_panel"].apply(lambda x: x.lower().replace(" ", "").replace("/", "") if x is not None else None)

    # Filter out samples that have already been processed in the files listed in args.resume_consider_list
    if args.resume_consider_list is not None and len(args.resume_consider_list) > 0:
        print(f"Filtering out samples that have already been processed in {len(args.resume_consider_list)} files")
        processed_row_ids = set()
        
        for file_path in args.resume_consider_list:
            try:
                # Read each file and collect the full_row_ids
                resume_df = pd.read_csv(file_path)
                if 'full_row_id' in resume_df.columns:
                    processed_row_ids.update(resume_df['full_row_id'].tolist())
                    print(f"Added {len(resume_df)} processed samples from {file_path}")
                else:
                    print(f"Warning: 'full_row_id' column not found in {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # Filter out the processed samples
        original_count = len(rel_df)
        rel_df = rel_df[~rel_df.index.isin(processed_row_ids)]
        filtered_count = len(rel_df)
        print(f"Filtered out {original_count - filtered_count} samples that were already processed")
        print(f"Remaining samples: {filtered_count}")

    # Filter out LRRC56 (not sure why this fails?)
    #rel_df = rel_df.loc[~(rel_df.hgnc_gene == "LRRC56"),:]
    print("rel_df", rel_df.shape)

    #import ipdb; ipdb.set_trace()

    # Subset - seed, etc:
    if args.vcep_list is not None:
        # Filter by VCEPS provided
        vcep_mask = rel_df.expert_panel.isin(args.vcep_list)
        rel_df = rel_df.loc[vcep_mask,:]
    elif args.date_lower_bound is not None:
        pdate_list = pd.to_datetime(rel_df['pub_date'], utc=True).dt.tz_localize(None)
        rel_df = rel_df.loc[(pdate_list > np.datetime64(args.date_lower_bound)),:]

    # Get pm mapper:
    pm_df = pd.read_csv(PMID_TEXT_PATH)

    if args.summarize_paper:
        #pm_df = pd.read_csv(PMID_TEXT_PATH)
        pm_summarized_df = pd.read_csv(PMID_SUMMARIZED_TEXT_PATH)

        # Create a mapper based on gene, diseas, and mode of inheritance:
        pm_summarized_df["unifying_row_mapper"] = pm_summarized_df["pmid"] + "_" + pm_summarized_df["variant"] + "_" + pm_summarized_df["disease"] + "_" + pm_summarized_df["inheritance"]
        pm_summarized_df.set_index("unifying_row_mapper", inplace=True)
        # Create a mapping dictionary from pmid to summary
        # summary_map = dict(zip(pm_summarized_df['pmid'], pm_summarized_df['summary']))
        # # Map summaries to pm_df using the dictionary
        # pm_df["full_text"] = pm_df["pmid"].map(summary_map)

        # Temp: check that all rel_df samples are in here:
        rel_df["unifying_row_mapper"] = rel_df["pmid"] + "_" + rel_df["variant"] + "_" + rel_df["disease"] + "_" + rel_df["mode_inheritance"]
        assert rel_df["unifying_row_mapper"].isin(pm_summarized_df.index).all(), "All rel_df samples must be in pm_summarized_df"

    input_template = EVIDENCE_PROMPT_WITH_LABEL


    pm_sub = pm_df.loc[pm_df.pmid.isin(rel_df.pmid)]
    pm_sub.set_index("pmid", inplace=True)

    # Now sample by numbers:
    if args.num_samples is not None:
        np.random.seed(args.seed)
        inds = np.random.choice(np.arange(rel_df.shape[0]), size=args.num_samples, replace=False)
        rel_df = rel_df.iloc[inds,:]

    # Now you know they'll all map - perform inference:

    minimal_result_dict = {
        "full_row_id": [],
        "vcep": [],
        "gene": [],
        "disease": [],
        "mondo_id": [],
        "pmid": [],
        "evidence_code": [],
        "met_status": [],
        "model_answer": [],
    }

    instruction_sys_prompt = INSTRUCTION_SYSTEM_PROMPT

    # Build ICL examples:
    icl_example_str = None
    if args.icl_shot_number is not None:
        
        icl_example_str = build_icl_examples(
            args, 
            icl_explanation = args.icl_explanation,
        )

        # Concat ICL examples:
        if args.icl_code_description:
            instruction_sys_prompt += "\n\nHere are some examples of how to score the evidence. \
                Examples of code descriptions are also given for these examples; however, you should continue to use the \
                codes given above to make your final determination about the evidence:\n" + icl_example_str
        else:
            instruction_sys_prompt += "\n\nHere are some examples of how to score the evidence:\n" + icl_example_str

    #import ipdb; ipdb.set_trace()

    # Set up LLM:
    if args.best_of_k > 1:
        model = BestofKAgent(
            system_prompt = instruction_sys_prompt,
            model_name = args.model_name,
            temperature = args.temperature,
            best_of_k = args.best_of_k,
        )
    else:
        model = OnePassAgent(
            system_prompt = instruction_sys_prompt,
            model_name = args.model_name,
            temperature = args.temperature,
        )

    total_cost = 0
    previous_intermediate_file = None
    
    # Iterate over sub_info_df:
    for i, row in tqdm(rel_df.iterrows(), total=rel_df.shape[0]):

        #evidence_codes = find_row_in_vcep_map(row, vcep_map).sort_values("label")
        evidence_codes = get_criteria_per_row(row, DATA_PATH, stack_descriptions = False)

        evidence_code_gt = row["evidence_code"].replace("-", "_").replace(" ", "")
        #try:
        gt_mask = (evidence_codes["aggregate_code"] == evidence_code_gt)
        if gt_mask.sum() == 0:
            print("ERROR: SKIPPING BECAUSE NO CODE IS FOUND")
            continue
        #try:
        evidence_code_desc = evidence_codes.loc[gt_mask, "Description"].values[0]
        # except:
        #     import ipdb; ipdb.set_trace()
            #evidence_code_desc = "No evidence code description found"

        #print("INPUT")
        #print(instruction_sys_prompt)

        evidence_code = row["evidence_code"]
        pmid = row["pmid"]
        variant = row["variant"]
        disease = row["disease"]
        inheritance = row["mode_inheritance"]
        abstract = pm_sub.loc[pmid, "abstract"]
        if args.summarize_paper:
            key = row["unifying_row_mapper"]

            if isinstance(pm_summarized_df.loc[key, "summary"], pd.Series):
                full_text = pm_summarized_df.loc[key, "summary"].iloc[0]
            else:
                full_text = pm_summarized_df.loc[key, "summary"]
        else:
            full_text = pm_sub.loc[pmid, "full_text"]

        if abstract is None:
            continue

        evidence_prompt = input_template.format(
            variant=variant,
            disease=disease,
            inheritance=inheritance,
            pmid=pmid,
            abstract=abstract,
            full_text=full_text,
            evidence_code=EVIDENCE_CODE_TEMPLATE.format(ecode=evidence_code_gt, desc=evidence_code_desc)
        )

        input_str = f"Now determine whether {evidence_code_gt} the following PubMed article related to the given variant, disease, and mode of inheritance:\n{evidence_prompt}"
        #input_str = f"Now determine whether {evidence_code_gt} is met given the following PubMed article related to the given variant, disease, and mode of inheritance:\n{evidence_prompt}"

        #import ipdb; ipdb.set_trace()

        #model.set_system_prompt(instruction_sys_prompt)
        max_tokens = 500 if "qwen" in args.model_name else 2000
        out, cost = model.forward(input_str, max_tokens=max_tokens, get_cost = True)
        print(f"OUT\n{out}")
        print(f"COST: ${cost}")
        total_cost += cost

        minimal_result_dict["full_row_id"].append(row.name)
        minimal_result_dict["vcep"].append(row["expert_panel"])
        minimal_result_dict["gene"].append(row["hgnc_gene"])
        minimal_result_dict["disease"].append(row["disease"])
        minimal_result_dict["mondo_id"].append(row["mondo_id"])
        minimal_result_dict["pmid"].append(pmid)
        minimal_result_dict["evidence_code"].append(evidence_code)
        minimal_result_dict["model_answer"].append(out)
        minimal_result_dict["met_status"].append(row["met_status"])
        
        # Save intermediate results every 25 steps
        if (i + 1) % 25 == 0:
            results_df = pd.DataFrame(minimal_result_dict)
            previous_intermediate_file = save_intermediate_results(results_df, args.save_path, i + 1, previous_intermediate_file)

    print("TOTAL COST: ${:.4f}".format(total_cost))
    results_df = pd.DataFrame(minimal_result_dict)
    results_df.to_csv(args.save_path, index=False)
    
    # Delete the final intermediate file if it exists
    if previous_intermediate_file and os.path.exists(previous_intermediate_file):
        try:
            os.remove(previous_intermediate_file)
            print(f"Deleted final intermediate file: {previous_intermediate_file}")
        except Exception as e:
            print(f"Warning: Could not delete final intermediate file {previous_intermediate_file}: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--dedup_filter", action="store_true", help="Use deduplicated samples")

    parser.add_argument("--abstract_only", action="store_true")
    parser.add_argument("--pmid_only", action="store_true")
    parser.add_argument("--summarize_paper", action="store_true", help = "Use summaries instead of full text for inputs")
    parser.add_argument("--save_path", type=str)

    parser.add_argument("--icl_shot_number", type=int, default=None)
    parser.add_argument("--icl_sample_seed", type=int, default=1234)
    parser.add_argument("--icl_explanation", action="store_true")
    parser.add_argument("--icl_code_description", action="store_true")

    parser.add_argument("--icl_omit_fulltext", action="store_true", help="Omit full text from the prompt")
    parser.add_argument("--icl_omit_paper", action="store_true", help="Omit the paper from the prompt")    
    parser.add_argument("--icl_summarize_paper", action="store_true", help="Use paper summaries instead of full text for ICL examples")

    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--best_of_k", type=int, default=1)

    parser.add_argument("--date_lower_bound", type=str, default=None)
    parser.add_argument('--vcep_list', nargs='*', help="VCEP list to consider", required=False, default=None)

    parser.add_argument('--resume_consider_list', nargs='*', help="files to consider for resumption", required=False, default=None)

    args = parser.parse_args()

    main(args)