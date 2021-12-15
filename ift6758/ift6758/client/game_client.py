import argparse
import json
import os
import requests
import pandas as pd
import logging
from utils import process_data


# setup basic logging configuration
logging_format = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"

logging.basicConfig(format=logging_format, level=logging.INFO)

logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self, ip: str = "127.0.0.1:5000", game_id: int = 2021020329, features=None):
        
        self.base_url = f"http://{ip}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        
        self.game_id = game_id
        logger.info(f"Game ID : {self.game_id}")
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
        
        print(pred_response_json)
        
        pred_dct = json.loads(pred_response_json)["predictions"]        
        pred_df = pd.DataFrame(pred_dct.values(), columns=['predictions'], index=X_data.index)

        return pred_df
    
    
    def retrieve_games(self):
        """
        Retrieve information for a single NHL Game IDs.
        Logs warnings if no data associated with game_id.
        """
        
        logger.info(f"Querying Game ID : {self.game_id}")

        url = f"https://statsapi.web.nhl.com/api/v1/game/{self.game_id}/feed/live/"

        response = requests.get(url)
        if response.status_code == 404:
            logging.info(f"No data returned for game ID: {game_id} (404)")
            return None
        else:
            return response.json()
    
    
def main(args):
    
    game_client = GameClient(ip = args.ip, game_id = args.game)
    file_path = f"./data/{args.game}.csv"
    pred_file_path = f"./data/{args.game}_pred.csv"
    if os.path.exists(file_path):
        if os.path.exists(pred_file_path):
            logger.info("Prediction exist")
        else:
            logger.info("Data file exist")
            X_test = pd.read_csv(file_path)
            print(X_test)
            y_preds = game_client.predict(X_test) 
            logging.info(f"predicting {len(y_preds)} events")
            y_preds.to_csv(pred_file_path)
            logger.info("Prediction complete")
    else:
        logger.info("Retrieving data")
        content = game_client.retrieve_games()
        
        X_test = process_data(game_id = args.game, data = content)
        y_preds = game_client.predict(X_test) 
        logging.info(f"predicting {len(y_preds)} events")
        y_preds.to_csv(pred_file_path)
        logger.info("Prediction complete")
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download NHL API Data and Make Prediction')
    
    parser.add_argument('-ip', '--ip', nargs='+', type=str,
                        default="127.0.0.1:5000",
                        help='Server Ip and port')  
    parser.add_argument('-g', '--game', nargs='+', type=int,
                        default=2021020329,
                        help='NHL game ID')   
    args = parser.parse_args()
    main(args)
    
    
