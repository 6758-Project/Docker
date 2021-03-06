import logging
import os
import requests
from typing import Callable

import pandas as pd

from .utils import parse_game_data

logger = logging.getLogger()


class UnknownGameException(Exception):
    pass


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

    def get_game_info(self, game_id: int) -> (pd.DataFrame, int):
        """ Returns a dataframe of shots from a given NHL game.

        Arguments:
            * game_id: the NHL game id
        Returns:
            * a DataFrame of shot events, including the results of preprocessing
        Raises:
            * UnknownGameException if a game_id is unrecognized
        """
        if game_id in self.loaded_games.keys():
            events = self.loaded_games[game_id]

        elif game_id in self.available_games:
            events = pd.read_csv(os.path.join(self.raw_game_directory, str(game_id)+".csv"))
            events = events.drop(columns=["Unnamed: 0"], errors='ignore')
        else:
            events = None

        previously_unseen_game = events is None
        incomplete_game = previously_unseen_game or ("Game End" not in list(events['description']))
        if previously_unseen_game or incomplete_game:
            new_events = self.query_api(game_id)

            necessary_to_update_cache = previously_unseen_game or (len(new_events) >= len(events))
            if necessary_to_update_cache:
                self.loaded_games[game_id] = new_events
                new_events.to_csv(os.path.join(self.raw_game_directory, str(game_id)+".csv"))

                events = new_events

        return events

    @staticmethod
    def query_api(game_id: int):
        """ Queries the NHL game feed API for GAME_ID """
        url = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/"
        response = requests.get(url)

        if response.status_code == 404:
            raise UnknownGameException("404: No data returned for game ID {game_id}")
        else:
            content = response.json()
            return parse_game_data(game_id=game_id, game_data=content)
