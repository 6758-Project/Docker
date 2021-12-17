import pandas as pd

from client.utils import get_preprocess_function, EVENT_COLS

class GameTracker:
    """ Generates a real-time NHL game tracker dashboard """

    def __init__(self, nhl_api_client: "NHLAPIClient", predictor_client: "PredictorAPIClient"):
        self.nhl_api_client = nhl_api_client
        self.predictor_client = predictor_client

        self.curr_game = None
        self.current_model = None
        self.events = pd.DataFrame(columns=EVENT_COLS)

    def update(self, game_id: int, model_id: str):
        """ Updates state for game_id with model_id predictions
        Raises: UnknownGameException, UnknownModelException
        """
        new_model = (self.current_model != model_id)
        new_game = (self.curr_game != game_id)

        game_events = self.nhl_api_client.get_game_info(game_id=game_id)

        if new_model:
            self.predictor_client.update_model(model_id)

        # only update if both API query and model update were successful
        self.curr_game = game_id
        self.current_model = model_id

        if new_model or new_game:
            self.events = pd.DataFrame(columns=EVENT_COLS)  # resets to empty

        preprocessed_events = get_preprocess_function(model_id)(game_events)

        new_pred_inputs = preprocessed_events.iloc[(len(self.events)+1):]
        # TODO make sure only shots are passed to predict -- then do intelligent join
        new_preds = self.predictor_client.predict(new_pred_inputs)

        new_events = pd.concat([game_events.iloc[(len(self.events)+1):], new_preds], axis=1)
        self.events = pd.concat([self.events, new_events])


    def get_events_and_predictions(self) -> pd.DataFrame:
        return self.events.iloc[::-1]
