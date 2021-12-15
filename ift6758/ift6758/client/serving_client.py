import json
import requests
import pandas as pd
import logging


# setup basic logging configuration
logging_format = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"

logging.basicConfig(format=logging_format, level=logging.INFO)

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        X_data_json = X_data.to_json()
        pred_response = requests.post(self.base_url + "/predict", json=X_data_json)
        pred_response_json = pred_response.json()

        pred_dct = json.loads(pred_response_json)["predictions"]
        pred_df = pd.DataFrame(
            pred_dct.values(), columns=["predictions"], index=X_data.index
        )

        return pred_df

    def get_logs(self) -> dict:
        """Get server logs"""
        logs_response = requests.get(self.base_url + "/logs")
        pretty_logs = json.dumps(logs_response.json(), indent=4, sort_keys=True)
        return pretty_logs

    def download_registry_model(
        self, workspace: str, model_name: str, version: str
    ) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it.

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model

        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        # "xgboost-feats-non-corr" is out best performer in from milestone 2
        if model_name != "xgboost-feats-non-corr":
            logging.warning(
                "Predict function might not work properly with the chosen model \
                the expected model name is 'xgboost-feats-non-corr'"
            )

        COMET_MODEL_INFO = {
            "project_name": "ift6758-hockey",
            "workspace": workspace,
            "model_type": "xgboost_non_corr",
            "model_desc": "XGBoost with Non Correlated Features",
            "model_name": model_name,
            "version": version,
            "file_name": "xgboost_feats_non_corr",
        }

        COMET_MODEL_INFO = json.dumps(COMET_MODEL_INFO)
        download_response = requests.post(
            self.base_url + "/download_registry_model", json=COMET_MODEL_INFO
        )
        return download_response.json()


if __name__ == "__main__":

    app_client = ServingClient()

    # getting the logs
    logs_dct = app_client.get_logs()
    logging.info(f"retrieved logs: {logs_dct}")

    # downloading model from comet
    download_res_msg = app_client.download_registry_model(
        workspace="tim-k-lee", model_name="xgboost-feats-non-corr", version="1.0.1"
    )
    logging.info(f"API response for download_registry_model: {download_res_msg}")

    # predict test data from milestone 2 (preprocessed for the best performer xgboost model)
    X_test = pd.read_csv("../data/offline_test_data.csv")
    y_preds = app_client.predict(X_test)
    logging.info(f"predicting {len(y_preds)} events")
