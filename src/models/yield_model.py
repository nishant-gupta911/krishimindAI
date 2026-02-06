"""Crop yield prediction model"""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from typing import Dict
import joblib


class YieldPredictor:
    """Multi-model ensemble for crop yield prediction"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42)
        }
        self.weights = {'rf': 0.3, 'gbm': 0.3, 'xgb': 0.4}
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train all models in ensemble"""
        for name, model in self.models.items():
            model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> float:
        """Ensemble prediction (weighted average)"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])
        return np.sum(predictions, axis=0)
    
    def save(self, path: str):
        """Save trained models"""
        joblib.dump(self.models, path)
    
    def load(self, path: str):
        """Load trained models"""
        self.models = joblib.load(path)
