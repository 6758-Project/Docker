import logging
import requests
from utils import process_data


# setup basic logging configuration
logging_format = "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"

logging.basicConfig(format=logging_format, level=logging.INFO)

logger = logging.getLogger(__name__)


class NHLAPIClient:
    """ A utility class for efficiently retrieving data from the NHL API """

    def __init__(self, model_type="xgboost_non_corr", raw_game_directory="./data"):
        self.model_type = model_type
        self.raw_game_directory = raw_game_directory
        # TODO add support for caching

    def get_game_info(self, game_id=2021020329, from_last_index=True):
        """
        Returns a dataframe of events for GAME_ID, starting FROM_LAST_INDEX if True
        Retrieve information for a single NHL Game IDs.
        Logs warnings if no data associated with game_id.
        """

        logger.info(f"Querying Game ID : {game_id}")

        url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"

        response = requests.get(url)
        if response.status_code == 404:
            logging.info(f"No data returned for game ID: {game_id} (404)")
            return None
        else:
            content = response.json()
            df = process_data(game_id=game_id, model_type=self.model_type, data=content)

            if from_last_index:
                output = df.iloc[::-1]
            else:
                output = df
            return output


if __name__ == "__main__":
    """For testing only"""
    game_client = NHLAPIClient()
    data = game_client.get_game_info()
    file_path = "./data/2021020329_T.csv"
    data.to_csv(file_path)
    data = game_client.get_game_info(from_last_index=False)
    file_path = "./data/2021020329_F.csv"
    data.to_csv(file_path)
