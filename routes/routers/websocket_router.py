from routes.controllers.training_controller import TrainingController
from routes.controllers.prediction_controller import PredictionController

websocket_router = {
    'train': TrainingController.train_subscribe,
    'predict': PredictionController.predict_subscribe,
}
