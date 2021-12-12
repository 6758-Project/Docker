import pandas as pd
import json
import requests


def test_predict_endpoint():
    X_test = pd.read_csv("offline_test_data.csv")
    X_test_json = X_test.to_json()

    pred_response = requests.post(
                        "http://127.0.0.1:5000/predict",
                        json=X_test_json)



def test_download_endpoint():
    COMET_MODEL_INFO = {
    "project_name": "ift6758-hockey",
    "workspace": "tim-k-lee",
    "model_type": "xgboost_non_corr",
    "model_desc": "XGBoost with Non Correlated Features",
    "model_name": "xgboost-feats-non-corr",
    "version": "1.0.1",
    "file_name": "xgboost_feats_non_corr",
    }

    COMET_MODEL_INFO = json.dumps(COMET_MODEL_INFO)
    download_response = requests.post(
                        "http://127.0.0.1:5000/download_registry_model",
                        json=COMET_MODEL_INFO)
    print(f"response: {download_response.json()}")


if __name__ == '__main__':

    # first time run the first line only
    test_download_endpoint()
    # test_predict_endpoint()

