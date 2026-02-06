"""Climate scenario simulation engine"""
import pandas as pd
from typing import Dict
from src.engine.recommender import CropRecommender


class ClimateSimulator:
    """Simulates impact of climate shocks on crop recommendations"""
    
    def __init__(self):
        self.recommender = CropRecommender()
    
    def simulate_rainfall_drop(self, district: str, season: str, year: int, 
                               drop_percent: float) -> Dict:
        """Simulate impact of rainfall reduction"""
        # Baseline recommendations
        baseline = self.recommender.rank_crops(district, season, year)
        
        # Adjust features for rainfall drop
        # (In production, modify weather features and re-predict)
        shocked = baseline.copy()
        for rec in shocked:
            rec['expected_yield_qtl_per_ha'] *= (1 - drop_percent / 100)
            rec['risk_score'] += drop_percent / 2
            rec['expected_revenue_per_ha'] = rec['expected_yield_qtl_per_ha'] * rec['expected_price_per_qtl']
        
        return {
            'scenario': f'Rainfall drop {drop_percent}%',
            'baseline': baseline,
            'shocked': shocked
        }
    
    def simulate_temperature_rise(self, district: str, season: str, year: int, 
                                  rise_celsius: float) -> Dict:
        """Simulate impact of temperature increase"""
        baseline = self.recommender.rank_crops(district, season, year)
        
        shocked = baseline.copy()
        for rec in shocked:
            rec['expected_yield_qtl_per_ha'] *= (1 - rise_celsius * 0.05)
            rec['risk_score'] += rise_celsius * 5
            rec['expected_revenue_per_ha'] = rec['expected_yield_qtl_per_ha'] * rec['expected_price_per_qtl']
        
        return {
            'scenario': f'Temperature rise {rise_celsius}°C',
            'baseline': baseline,
            'shocked': shocked
        }
