import pandas as pd
import os

def convert_row_code(row):
    code, strength = row["Code"], row["Strength"]

    if strength.startswith("Original ACMG"):
        return code
    else:
        return f"{code}_{strength.replace(' ', '')}"

def cascade_descriptions_fn(criteria_df):

    modify_criteria_df = criteria_df.copy()

    for code in modify_criteria_df["Code"].unique():
        in_code = modify_criteria_df.loc[modify_criteria_df["Code"] == code]

        mask_in = in_code["Strength"].apply(lambda x: x.startswith("Original ACMG"))
        if not (mask_in.sum() == 1):
            print(in_code)
            raise ValueError

        org_row = in_code[mask_in].iloc[0]
        org_desc = org_row["Description"]

        for i in range(in_code.shape[0]):
            strength = in_code["Strength"].iloc[i]
            if not strength.startswith("Original ACMG"):
                new_description = "General code description: " + org_desc + "\n\n" + "Detailed code description: " + in_code["Description"].iloc[i]
            else:
                new_description = "General code description: " + org_desc
                
            modify_criteria_df.loc[
                (modify_criteria_df["Code"] == code) & (modify_criteria_df["Strength"] == strength),
                "Description"
            ] = new_description

    return modify_criteria_df

def stack_descriptions_fn(criteria_df):

    modify_criteria_df = criteria_df.copy()

    for code in modify_criteria_df["Code"].unique():
        in_code = modify_criteria_df.loc[modify_criteria_df["Code"] == code]

        mask_in = in_code["Strength"].apply(lambda x: x.startswith("Original ACMG"))
        if not (mask_in.sum() == 1):
            print(in_code)
            raise ValueError

        org_row = in_code[mask_in].iloc[0]
        org_desc = org_row["Description"]

        for i in range(in_code.shape[0]):
            strength = in_code["Strength"].iloc[i]
            if not strength.startswith("Original ACMG"):
                new_description = "Detailed code description: " + in_code["Description"].iloc[i]
            else:
                new_description = "General code description: " + org_desc
                
            modify_criteria_df.loc[
                (modify_criteria_df["Code"] == code) & (modify_criteria_df["Strength"] == strength),
                "Description"
            ] = new_description

    return modify_criteria_df
    

def get_criteria_per_row(row, DATA_PATH, stack_descriptions = False):
    base_path = row["path"]
    criteria = pd.read_csv(os.path.join(DATA_PATH, "VCI/parsing_csr_criteria/version_csv_individual", base_path))
    if criteria["Gene"].unique().shape[0] == 1:
        # Aggregate codes:
        criteria["aggregate_code"] = [convert_row_code(row) for i, row in criteria.iterrows()]
        if stack_descriptions:
            return stack_descriptions_fn(criteria)[["aggregate_code", "Description"]]
        else:
            return cascade_descriptions_fn(criteria)[["aggregate_code", "Description"]]
    else:     
        #print("HERE")
        criteria_trim = criteria.loc[criteria["Gene"].apply(lambda x: x.split(" ")[0]) == row["hgnc_gene"],:].copy()
        #criteria_trim["aggregate_code"] = [convert_row_code(row) for i, row in criteria_trim.iterrows()]
        criteria_trim["aggregate_code"] = criteria_trim.apply(convert_row_code, axis=1)
        #return cascade_descriptions(criteria_trim)[["aggregate_code", "Description"]]
        if stack_descriptions:
            return stack_descriptions_fn(criteria_trim)[["aggregate_code", "Description"]]
        else:
            return cascade_descriptions_fn(criteria_trim)[["aggregate_code", "Description"]]

