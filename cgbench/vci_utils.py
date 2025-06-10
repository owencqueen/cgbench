import os, sys
import pandas as pd

def build_vcep_map(
        vcep_directory: str, 
        vcep_csv_dirpath: str,
    ):

    # Load each vcep:
    vcep_transform = lambda x: x.replace(" Variant Curation Expert Panel", "").replace(" ", "").replace("/", "").lower()
    vcep_map = {vcep_transform(v):dict() for v in vcep_directory.vcep.unique()}

    for _, row in vcep_directory.iterrows():

        did = "any"
        gid = "any"
        if row.isna()["gene"]:
            pass
        elif row.isna()["disease_id"]: # May not be conditional on disease
            gid = row["gene"]
            did = "any"
        else:
            gid = row["gene"]
            did = row["disease_id"]
            #key = (row["gene"], did)
        vcep_df = pd.read_csv(os.path.join(vcep_csv_dirpath, row["criteria_path"]))

        if gid in vcep_map[vcep_transform(row["vcep"])].keys():
            vcep_map[vcep_transform(row["vcep"])][gid].update({did: vcep_df})
        else:
            vcep_map[vcep_transform(row["vcep"])][gid] = {did: vcep_df}


    return vcep_map

def find_row_in_vcep_map(row, vcep_map):
    map1 = vcep_map[row["expert_panel"]]

    df = None
    if row["hgnc_gene"] in map1.keys():
        if row["mondo_id"] in map1[row["hgnc_gene"]].keys():
            df = map1[row["hgnc_gene"]][row["mondo_id"]]
        elif "any" in map1[row["hgnc_gene"]].keys():
            df = map1[row["hgnc_gene"]]["any"]
        else:
            pass
    else:
        if "any" in map1.keys():
            df = map1["any"]["any"]
        else:
            pass
    return df