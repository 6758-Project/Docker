""" 
Read and preprocessing input data based on selected model.
"""

import pandas as pd
from utils import INFREQUENT_STOPPAGE_EVENTS, LABEL_COL, TRAIN_COLS_PART_4


# Load data from the specific file path, select all, regular season, or post season data.
def load_dataset(file_path):

    df = pd.read_csv(file_path)
    df["game_id"] = df["game_id"].astype(int)

    X_data = df[TRAIN_COLS_PART_4]
    Y_data = df[LABEL_COL].astype(int)

    return X_data, Y_data


def process_data(test_path):

    # Retrieve and process data and model
    X_Test, Y_Test = load_dataset(test_path)

    X_processed = XGB_Non_Corr_preprocess(X_Test)
    return X_processed, Y_Test


def XGB_Non_Corr_preprocess(X_data):

    X_data["secondary_type"].replace({"Tip-in": "Deflection"}, inplace=True)
    X_data["prev_event_type"].replace(
        to_replace=INFREQUENT_STOPPAGE_EVENTS, value="STOP", inplace=True
    )

    X_data = pd.get_dummies(X_data, ["shot", "prev_event"])

    #    the redundant features after inspecting them in "./notebooks/M2_detect-feat-correlation.ipynb"
    redundant_feats = ["is_rebound", "coordinate_y", "coordinate_x", "period"]

    # Training and validation data of the selected features
    selected_feats = X_data.columns.difference(redundant_feats)
    print(selected_feats)
    X_data = X_data[selected_feats]

    return X_data


if __name__ == "__main__":

    X_test, Y_test = load_dataset(file_path="./data/test_processed.csv")
    X_test_processed = preprocess_data(X_test)
    X_test_processed.to_csv("./data/offline_test_data.csv")

