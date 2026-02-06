"""FastAPI application for KrishiMind AI"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from src.engine.recommender import CropRecommender
from src.engine.simulator import ClimateSimulator
from src.config import API_HOST, API_PORT

app = FastAPI(
    title="KrishiMind AI API",
    description="AI Crop Planning & Resource Optimization Engine",
    version="1.0.0"
)

recommender = CropRecommender()
simulator = ClimateSimulator()


class RecommendationRequest(BaseModel):
    district: str
    season: str
    year: int
    top_n: int = 3


class SimulationRequest(BaseModel):
    district: str
    season: str
    year: int
    scenario_type: str  # "rainfall_drop" or "temperature_rise"
    magnitude: float


@app.get("/")
def root():
    return {"message": "KrishiMind AI API", "status": "active"}


@app.post("/api/v1/recommend")
def get_recommendations(request: RecommendationRequest) -> Dict:
    """Get top-N crop recommendations for district and season"""
    try:
        recommendations = recommender.rank_crops(
            request.district, 
            request.season, 
            request.year, 
            request.top_n
        )
        return {
            "district": request.district,
            "season": request.season,
            "year": request.year,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/simulate")
def run_simulation(request: SimulationRequest) -> Dict:
    """Run climate scenario simulation"""
    try:
        if request.scenario_type == "rainfall_drop":
            result = simulator.simulate_rainfall_drop(
                request.district, request.season, request.year, request.magnitude
            )
        elif request.scenario_type == "temperature_rise":
            result = simulator.simulate_temperature_rise(
                request.district, request.season, request.year, request.magnitude
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid scenario type")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
