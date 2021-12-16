import json
import logging
import requests

import pandas as pd

from .utils import AVAILABLE_MODELS


# setup basic logging configuration
logging_format = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictorAPIClient:
    """ A convenience interface for interacting with the Prediction API """

    def __init__(self, ip: str = "0.0.0.0", port: int = 5000):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        self.curr_comet_model_name = None

    def predict(self, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            x_data (Dataframe): Input dataframe to submit to the prediction service.
        """
        x_data_json = x_data.to_json()
        pred_response = requests.post(self.base_url + "/predict", json=x_data_json)
        pred_response_json = pred_response.json()

        pred_dct = json.loads(pred_response_json)["predictions"]
        pred_df = pd.DataFrame(
            pred_dct.values(), columns=["predictions"], index=x_data.index
        )

        return pred_df

    def get_logs(self) -> dict:
        """Get server logs"""
        logs_response = requests.get(self.base_url + "/logs")
        pretty_logs = json.dumps(logs_response.json(), indent=4, sort_keys=True)

        return pretty_logs

    def update_model(
        self, comet_model_name: str = "xgboost-feats-non-corr", workspace: str = "tim-k-lee", version: str = None
    ) -> dict:
        """
        Triggers a 'model swap' in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it.

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model

        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download (defaults to most-recent)
        """

        if comet_model_name not in AVAILABLE_MODELS.keys():
            logging.exception(
                f"{comet_model_name} doesn't exist. Available models are: {AVAILABLE_MODELS.keys()}"
            )
            logging.info(f"model is still {self.curr_comet_model_name}")

            return None
        else:
            comet_model_info = AVAILABLE_MODELS[comet_model_name]
            comet_model_info["workspace"] = workspace
            if version is not None:
                comet_model_info["version"] = version

            comet_model_info_json = json.dumps(comet_model_info)
            response = requests.post(
                self.base_url + "/download_registry_model", json=comet_model_info_json
            )

            self.curr_comet_model_name = comet_model_name

        return response.json()




if __name__ == "__main__":

    app_client = PredictorAPIClient()

    download_res_msg = app_client.update_model()
    logging.info(f"API response for update_model: {download_res_msg}")

    # predict test data from milestone 2
    x_test = pd.read_csv("../data/test_processed.csv")
    y_preds = app_client.predict(x_test)
    logging.info(f"predicting {len(y_preds)} events")

    app_client.update_model("nn-adv")

    # predict test data from milestone 2
    x_test = pd.read_csv("../data/test_processed.csv")
    y_preds = app_client.predict(x_test)
    logging.info(f"predicting {len(y_preds)} events")

    # getting the logs
    logs_dct = app_client.get_logs()
    logging.info(f"retrieved logs: {logs_dct}")
