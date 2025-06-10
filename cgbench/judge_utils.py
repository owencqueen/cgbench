import pandas as pd

def get_VCI_comments_from_results_df(
        results_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        col_name = "summary_comments",
    ) -> pd.DataFrame:
    return test_df[col_name].iloc[results_df["full_row_id"]]