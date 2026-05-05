from src.models.base import BaseModel, ModelResult
from src.models.linear import (
    OLSModel, RidgeModel, LassoModel, ElasticNetModel,
)
from src.models.trees import DecisionTreeModel, RandomForestModel
from src.models.boosting import LightGBMModel
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMModel
from src.models.mlp import MLPModel

MODEL_REGISTRY = {
    "ols": OLSModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "elasticnet": ElasticNetModel,
    "decision_tree": DecisionTreeModel,
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "arima": ARIMAModel,
    "lstm": LSTMModel,
    "mlp": MLPModel,
}

__all__ = [
    "BaseModel", "ModelResult", "MODEL_REGISTRY",
    "OLSModel", "RidgeModel", "LassoModel", "ElasticNetModel",
    "DecisionTreeModel", "RandomForestModel", "LightGBMModel",
    "ARIMAModel", "LSTMModel", "MLPModel",
]
