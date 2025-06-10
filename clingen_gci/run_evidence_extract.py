import argparse, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Optional

from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, Type, TypeVar

sys.path.append("/home/users/oqueen/clingen_benchmark")

from cgbench.util_agents import OnePassAgent, MODEL_API_MAP
from cgbench.prompts.gci_experimental_evidence import *
from cgbench.gci_utils import sop_to_lists, parse_evidence_text

DATA_PATH = "/home/users/oqueen/clingen_benchmark/data"
EVIDENCE_TRAIN_PATH = os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/train.csv")
EVIDENCE_TRAIN_DATESPLIT_PATH = os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/train_datesplit.csv")
EVIDENCE_TEST_PATH = os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/test.csv")
EVIDENCE_TEST_DATESPLIT_PATH = os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/test_datesplit.csv")
ORIGINAL_EE_PATH = os.path.join(DATA_PATH, "GCI/Clingen-Gene-Disease-Summary-2025-03-31.csv")

PUBMED_DF_PATH = os.path.join(DATA_PATH, "GCI/pubmed/experimental_evidence.csv")

T = TypeVar('T')

def dict_to_dataclass(data: Dict[str, Any], class_name: str = "DynamicDataclass") -> Type[T]:
    """
    Convert a dictionary into a dataclass where each key becomes a field.
    
    Args:
        data (Dict[str, Any]): The dictionary to convert
        class_name (str, optional): Name for the generated dataclass. Defaults to "DynamicDataclass".
    
    Returns:
        Type[T]: A new dataclass type with fields matching the dictionary keys
        
    Example:
        >>> data = {"name": "John", "age": 30}
        >>> Person = dict_to_dataclass(data, "Person")
        >>> person = Person(**data)
        >>> print(person.name)  # Output: John
        >>> print(person.age)   # Output: 30
    """
    # Create fields for the dataclass
    fields = [(key, type(value)) for key, value in data.items()]
    
    # Create and return the dataclass
    return make_dataclass(class_name, fields)

def print_evidence_output(evidence_output_list: BaseModel):
    """Print each field of an ExperimentalEvidenceOutput object on a separate line.
    
    Args:
        evidence_output: A pydantic BaseModel object containing experimental evidence fields
    """
    #import ipdb; ipdb.set_trace()
    for i, evidence_output in enumerate(evidence_output_list.evidence_list):
        print("EVIDENCE {}".format(i+1))
        print(f"Experimental Category '{evidence_output.experimental_category}'")
        print(f"Explanation: '{evidence_output.explanation}'") 
        print(f"Score: {evidence_output.score}")
        if evidence_output.score_adjustment_reason:
            print(f"Score Adjustment Reason: '{evidence_output.score_adjustment_reason}'")

def print_evidence_output_str(parsed_evidence_out):
    """Print each field of an ExperimentalEvidenceOutput object on a separate line.
    
    Args:
        evidence_output: A pydantic BaseModel object containing experimental evidence fields
    """
    #import ipdb; ipdb.set_trace()
    for i, evidence_output in enumerate(parsed_evidence_out):
        print("\n")
        print("EVIDENCE {}".format(i+1))
        print(f"Experimental Category '{evidence_output['category']}'")
        print(f"Explanation: '{evidence_output['explanation']}'") 
        print(f"Score: {evidence_output['score']}")
        if 'score_adjustment_reason' in evidence_output.keys():
            print(f"Score Adjustment Reason: '{evidence_output['score_adjustment_reason']}'")

def make_icl_example(ee_row, pm_row, edf, example_template):


    extracted_evidence = ""
    for i, edf_row in edf.iterrows():
        extracted_evidence += "<evidence>"

        #import ipdb; ipdb.set_trace()

        if edf_row.isna()["Reason for Changed Score"]:
            sar = "None - default score assigned"
        else:
            sar = edf_row["Reason for Changed Score"]

        extracted_evidence += ONE_STRUCTURED_OUTPUT_PROMPT.format(
            category = edf_row["Experimental Category"],
            explanation = edf_row["Explanation"],
            score = edf_row["Points (default points)"].split(" ")[0],
            score_adjustment_reason = sar,
        )
        extracted_evidence += "</evidence>\n"

    input_str = example_template.format(
        gene = ee_row["GENE SYMBOL"],
        disease = ee_row["DISEASE LABEL"],
        mode_of_inheritance = ee_row["MOI"],
        pmid = pm_row.name,
        abstract = pm_row["abstract"],
        full_text = pm_row["full_text"].replace("\n", " "),
        extracted_evidence = extracted_evidence,
    )

    return input_str


def build_icl_examples(args):
    '''
    Make ICL examples based on the training data and desired ICL parameters
    '''

    original_ee_df = pd.read_csv(ORIGINAL_EE_PATH)

    example_template = None
    if args.icl_omit_fulltext:
        example_template = EVIDENCE_EXTRACTION_LABEL_TEMPLATE_OMIT_FULLTEXT
    elif args.icl_omit_paper:
        example_template = EVIDENCE_EXTRACTION_LABEL_TEMPLATE_OMIT_PAPER
    else:
        example_template = EVIDENCE_EXTRACTION_LABEL_TEMPLATE
        

    if args.original_split:
        train_examples = pd.read_csv(os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/train.csv"))
    else:
        train_examples = pd.read_csv(os.path.join(DATA_PATH, "GCI/evidence_tables/experimental_evidence/train_datesplit.csv"))

    # There are so many that we'll filter to ones with non-empty explanations:
    train_examples = train_examples[train_examples["Explanation"].notna()]

    # Try to achieve balance in codes when sampling:
    tc_vc = train_examples["Experimental Category"].value_counts()
    train_ec_all = tc_vc.index # Sorts into most common codes
    ncat = len(train_ec_all)
    train_ec = train_ec_all.tolist()[:args.icl_shot_number]
    if len(train_ec) < args.icl_shot_number:
        tc_vc -= 1
        n_times_add = args.icl_shot_number // ncat
        for _ in range(n_times_add - 1):
            if (tc_vc <= 0).any():
                # Need to mask:
                mask = (tc_vc <= 0)
                train_ec_all = train_ec_all[~mask]
            train_ec.extend(train_ec_all.tolist())
            tc_vc -= 1
        train_ec.extend(train_ec_all.tolist()[:(args.icl_shot_number % ncat)])

    # Get value counts of experimental categories in train_ec
    vc_ecsamples = pd.Series(train_ec).value_counts()
    
    # Initialize empty list to store sampled rows
    sampled_rows = []
    used_pairs = set()  # Track used (primary_index, pmid) pairs
    
    # For each experimental category, sample the specified number of rows
    for exp_cat, n_samples in vc_ecsamples.items():
        # Get rows matching this experimental category
        category_rows = train_examples[train_examples["Experimental Category"] == exp_cat]
        
        # Filter out rows with already used primary_index/pmid pairs
        available_rows = category_rows[~category_rows.apply(lambda x: (x["primary_index"], x["pmid"]) in used_pairs, axis=1)]
        
        # Sample n rows with replacement if needed
        n_available = len(available_rows)
        if n_available == 0:
            continue
            
        n_to_sample = min(n_samples, n_available)
        sampled = available_rows.sample(n=n_to_sample, random_state=args.seed)
        
        # Update used pairs
        for _, row in sampled.iterrows():
            used_pairs.add((row["primary_index"], row["pmid"]))
            
        sampled_rows.append(sampled)
    
    # Combine all sampled rows into single dataframe
    train_examples_sampled = pd.concat(sampled_rows)

    # Load pmdf:
    pm_df = pd.read_csv(PUBMED_DF_PATH)
    pm_df.set_index("pmid", inplace=True)

    # Get all rows in train_examples where "Experimental Category" is in train_ec
    gb_evidence_df = train_examples_sampled.groupby(["primary_index", "pmid"])
    tlen = len(list(gb_evidence_df))

    icl_string = ""

    counter = 1
    for (primary_index, pmid), _ in gb_evidence_df:

        mask_in = (train_examples["primary_index"] == primary_index) & (train_examples["pmid"] == pmid)
        full_evidence_df_i = train_examples[mask_in]

        #print("FULL EVIDENCE DF SHAPE: {}".format(full_evidence_df_i.shape))

        row_in_ee = original_ee_df.iloc[primary_index]
        pm_row = pm_df.loc[pmid]

        # Construct inputs:
        input_str = make_icl_example(
            ee_row = row_in_ee, 
            pm_row = pm_row, 
            edf = full_evidence_df_i,
            example_template = example_template
        )

        # Append to icl_string
        icl_string += "Example {}:".format(counter)
        icl_string += input_str
        icl_string += "\n"

        counter += 1

    return "Here are some examples of evidence extractions:\n" + icl_string

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

    # Load data from main path:
    original_ee_df = pd.read_csv(ORIGINAL_EE_PATH)

    if args.original_split:
        evidence_df = pd.read_csv(EVIDENCE_TEST_PATH)
    else:
        evidence_df = pd.read_csv(EVIDENCE_TEST_DATESPLIT_PATH)

    # Filter out samples that have already been processed in the files listed in args.resume_consider_list
    if args.resume_consider_list is not None and len(args.resume_consider_list) > 0:
        print(f"Filtering out samples that have already been processed in {len(args.resume_consider_list)} files")
        processed_row_ids = set()
        
        for file_path in args.resume_consider_list:
            try:
                # Read each file and collect the full_row_ids
                resume_df = pd.read_csv(file_path)
                if 'primary_index' in resume_df.columns and 'pmid' in resume_df.columns:
                    # Create tuples of (primary_index, pmid) for processed samples
                    processed_samples = list(zip(resume_df['primary_index'], resume_df['pmid']))
                    processed_row_ids.update(processed_samples)
                    print(f"Added {len(resume_df)} processed samples from {file_path}")
                else:
                    print(f"Warning: Required columns not found in {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # Filter out the processed samples
        original_count = len(evidence_df)
        evidence_df = evidence_df[~evidence_df.apply(lambda x: (x['primary_index'], x['pmid']) in processed_row_ids, axis=1)]
        filtered_count = len(evidence_df)
        print(f"Filtered out {original_count - filtered_count} samples that were already processed")
        print(f"Remaining samples: {filtered_count}")

    # Sample if needed:
    if args.sample_size is not None:
        # Sample to PMIDs
        evidence_df_pmid = evidence_df["pmid"].sample(n=args.sample_size, random_state=args.seed)
        evidence_df = evidence_df[evidence_df["pmid"].isin(evidence_df_pmid)]

    gb_evidence_df = evidence_df.groupby(["primary_index", "pmid"])

    # gb_list = list(gb_evidence_df)

    # import ipdb; ipdb.set_trace()

    pm_df = pd.read_csv(PUBMED_DF_PATH)

    pm_sub = pm_df.loc[pm_df.pmid.isin(evidence_df.pmid)]
    pm_sub.set_index("pmid", inplace=True)

    minimal_result_dict = {
        "primary_index": [],
        "pmid": [],
        "gene": [],
        "disease": [],
        "mondo_id": [],
        "model_answer": [],
        "parsing_error": []
    }

    sop_path_gen = lambda sop_num: os.path.join(DATA_PATH, "GCI/SOP/experimental_evidence/{}.json".format(sop_num))

    total_cost = 0

    tlen = len(list(gb_evidence_df))

    # Set up LLM:
    model = OnePassAgent(
        system_prompt = EVIDENCE_EXTRACTION_SYSTEM_PROMPT,
        model_name = args.model_name,
        temperature = args.temperature,
    )

    if args.icl_shot_number is not None:
        icl_examples = build_icl_examples(args)


    previous_intermediate_file = None
    for i, ((primary_index, pmid), evidence_df_i) in enumerate(tqdm(gb_evidence_df, total = tlen)):
        row_in_ee = original_ee_df.iloc[primary_index]
        #pmid = row["pmid"]

        sop_titles, sop_descriptions = sop_to_lists(sop_path_gen(row_in_ee["SOP"]))
        #struct_output_class = ee_structured_output_with_category_options(sop_titles, sop_descriptions, model_name = args.model_name)
        struct_prompt = ee_string_prompt_with_category_options(sop_titles, sop_descriptions)

        # Make input:
        evidence_prompt = EVIDENCE_EXTRACTION_INPUT_TEMPLATE.format(
            gene = row_in_ee["GENE SYMBOL"],
            disease = row_in_ee["DISEASE LABEL"],
            mode_of_inheritance = row_in_ee["MOI"],
            pmid = pmid,
            abstract = pm_sub.loc[pmid, "abstract"],
            full_text = pm_sub.loc[pmid, "full_text"].replace("\n", " "),
        )

        if args.icl_shot_number is not None:
            input_str = struct_prompt + "\n" + icl_examples + "\n" + f"Now extract evidence from the following PubMed article related to the given gene, disease, and mode of inheritance:{evidence_prompt}"
        else:   
            input_str = struct_prompt + "\n" + f"Now extract evidence from the following PubMed article related to the given gene, disease, and mode of inheritance:{evidence_prompt}"

        out, cost = model.forward(
            input = input_str,
            #response_format = struct_output_class,
            get_cost = True,
        )

        parsed_out = parse_evidence_text(out)
        print("OUT")
        if parsed_out == -1:
            print("PARSING ERROR")
            #continue
        else:
            print_evidence_output_str(parsed_out)

        # if MODEL_API_MAP[args.model_name] == "ANTHROPIC":
        #     out = make_dataclass("AnthropicOutput", [("evidence_list", [dict_to_dataclass(d) for d in out["evidence_list"]])])
        print(f"COST: ${cost}")

        #import ipdb; ipdb.set_trace()

        total_cost += cost

        #for evidence_output in out.evidence_list:
        
        minimal_result_dict["primary_index"].append(primary_index.item())
        minimal_result_dict["pmid"].append(pmid)
        minimal_result_dict["gene"].append(row_in_ee["GENE SYMBOL"])
        minimal_result_dict["disease"].append(row_in_ee["DISEASE LABEL"]) 
        minimal_result_dict["mondo_id"].append(row_in_ee["DISEASE ID (MONDO)"])
        #minimal_result_dict["experimental_category"].append(row["Experimental Category"])
        minimal_result_dict["model_answer"].append(out)
        minimal_result_dict["parsing_error"].append(parsed_out == -1)

        # Save intermediate results every 25 steps
        if (i + 1) % 25 == 0 and args.save_path is not None:
            results_df = pd.DataFrame(minimal_result_dict)
            previous_intermediate_file = save_intermediate_results(results_df, args.save_path, i + 1, previous_intermediate_file)
            print("Parsing errors so far: {}".format(np.sum(minimal_result_dict["parsing_error"])))

    print("TOTAL COST: ${:.4f}".format(total_cost))
    print("Final parsing errors: {}".format(np.sum(minimal_result_dict["parsing_error"])))
    results_df = pd.DataFrame(minimal_result_dict)
    if args.save_path is not None:
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
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default = 0.2)

    parser.add_argument("--original_split", type=bool, default = False)
    parser.add_argument("--sample_size", type=int, default = None)
    parser.add_argument("--seed", type=int, default = 1235)

    parser.add_argument("--icl_shot_number", type=int, default = None)
    parser.add_argument("--icl_omit_fulltext", action="store_true", help="Omit full text from the prompt")
    parser.add_argument("--icl_omit_paper", action="store_true", help="Omit the paper from the prompt")
    parser.add_argument("--icl_summarize_paper", action="store_true", help="Use paper summaries instead of full text for ICL examples")

    parser.add_argument("--save_path", type=str, default = None)
    parser.add_argument('--resume_consider_list', nargs='*', help="files to consider for resumption", required=False, default=None)

    args = parser.parse_args()

    main(args)