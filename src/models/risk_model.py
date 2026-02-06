"""Climate risk scoring model"""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib


class RiskScorer:
    """Calculates climate risk score (0-100) for crop-district combination"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_factors = ['rainfall_variability', 'temp_extremes', 'drought_history']
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train risk classification model"""
        self.model.fit(X, y)
    
    def predict_risk_score(self, X: pd.DataFrame) -> float:
        """Returns risk score 0-100 (higher = riskier)"""
        proba = self.model.predict_proba(X)
        # Convert probability to risk score
        risk_score = proba[:, 1] * 100 if proba.shape[1] > 1 else proba[:, 0] * 100
        return risk_score
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
