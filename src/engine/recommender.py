"""Crop recommendation and ranking engine"""
import pandas as pd
from typing import List, Dict
from src.models.yield_model import YieldPredictor
from src.models.price_model import PricePredictor
from src.models.risk_model import RiskScorer
from src.data.ingestion import DataIngestion
from src.config import MAJOR_CROPS, MODEL_DIR


class CropRecommender:
    """Decision engine for district-level crop recommendations"""
    
    def __init__(self):
        self.yield_model = YieldPredictor()
        self.price_model = PricePredictor()
        self.risk_model = RiskScorer()
        self.data_loader = DataIngestion()
        
        # Load trained models
        self.yield_model.load(MODEL_DIR / "yield_model.joblib")
        self.price_model.load(MODEL_DIR / "price_model.joblib")
        self.risk_model.load(MODEL_DIR / "risk_model.joblib")
    
    def calculate_revenue(self, yield_per_ha: float, price_per_quintal: float) -> float:
        """Calculate expected revenue per hectare"""
        return yield_per_ha * price_per_quintal
    
    def rank_crops(self, district: str, season: str, year: int, top_n: int = 3) -> List[Dict]:
        """Generate top-N ranked crop recommendations"""
        recommendations = []
        
        for crop in MAJOR_CROPS:
            # Prepare features
            features = self.data_loader.prepare_features(district, crop, season, year)
            
            # Predictions
            yield_pred = self.yield_model.predict(features)[0]
            price_pred = self.price_model.predict(features)[0]
            risk_score = self.risk_model.predict_risk_score(features)[0]
            
            # Calculate revenue
            revenue = self.calculate_revenue(yield_pred, price_pred)
            
            # Composite score (revenue weighted by risk)
            score = revenue * (1 - risk_score / 100)
            
            recommendations.append({
                'crop': crop,
                'expected_yield_qtl_per_ha': round(yield_pred, 2),
                'expected_price_per_qtl': round(price_pred, 2),
                'expected_revenue_per_ha': round(revenue, 2),
                'risk_score': round(risk_score, 2),
                'composite_score': round(score, 2)
            })
        
        # Sort by composite score and return top N
        recommendations.sort(key=lambda x: x['composite_score'], reverse=True)
        return recommendations[:top_n]
