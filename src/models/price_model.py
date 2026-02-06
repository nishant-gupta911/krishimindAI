"""Crop price prediction model"""
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import joblib


class PricePredictor:
    """Predicts future crop prices based on historical mandi data"""
    
    def __init__(self):
        self.model = LGBMRegressor(n_estimators=100, random_state=42)
        self.trend_model = Ridge(alpha=1.0)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train price prediction model"""
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> float:
        """Predict crop price per quintal"""
        return self.model.predict(X)
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
