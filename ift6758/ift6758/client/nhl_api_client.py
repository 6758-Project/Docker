import logging
import os
import requests
from typing import Callable

import pandas as pd

from .utils import parse_game_data

logger = logging.getLogger()

class NHLAPIClient:
    """ A utility class for efficiently retrieving data from the NHL API """

    # TODO update raw_game_directory to internal container data dir
    def __init__(self, raw_game_directory="./data"):
        self.raw_game_directory = raw_game_directory

        if not os.path.exists(self.raw_game_directory):
            logger.warning("provided game_directory did not exist, so creating it.")
            os.makedirs(self.raw_game_directory)

        self.available_games = [f for f in os.listdir(self.raw_game_directory) if f.endswith(".csv")]
        self.loaded_games = {}

    def get_game_info(
        self,
        game_id: int,
        preprocess_func: Callable = None,
        incremental_only: bool = True
    ) -> pd.DataFrame:
        """ Returns a dataframe of shots from a given NHL game.

        Arguments:
            * game_id: the NHL game id
            * preprocess_func: (optional) a function to apply to the dataframe before returning
            * incremental_only: whether or not to return only previously-unreturned shots

        Returns:
            * a DataFrame of shot events, including the results of preprocessing
        """
        previous_returned_idx = -1

        if game_id in self.loaded_games.keys():
            events = self.loaded_games[game_id]
            previous_returned_idx = events['event_index'].max()

        elif game_id in self.available_games:
            events = pd.read_csv(os.path.join(self.raw_game_directory, str(game_id)+".csv"))
        else:
            events = None

        previously_unseen_game = events is None
        incomplete_game = previously_unseen_game or ("Game End" not in list(events['description']))

        if previously_unseen_game or incomplete_game:
            new_events = self.query_api(game_id)

            if previously_unseen_game or (len(new_event) >= len(events)):
                self.loaded_games[game_id] = new_events
                new_events.to_csv(os.path.join(self.raw_game_directory, str(game_id)+".csv"))

                events = new_events

        events = preprocess_func(events)

        if incremental_only:
            events = events[events['event_index']>previous_returned_idx]

        return events

    @staticmethod
    def query_api(game_id: int):
        url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"
        response = requests.get(url)

        if response.status_code == 404:
            raise ValueError("404: No data returned for game ID {game_id}")
        else:
            content = response.json()
            return parse_game_data(game_id=game_id, game_data=content)
