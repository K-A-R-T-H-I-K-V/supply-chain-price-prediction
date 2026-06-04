"""
MCP (Model Context Protocol) Weather Server
Provides real-time weather data for supply chain predictions

This module reads optional environment variables to override the default
Open-Meteo endpoints:
- WEATHER_API_URL (default: https://api.open-meteo.com/v1/forecast)
- GEOCODING_API_URL (default: https://geocoding-api.open-meteo.com/v1/search)

Open-Meteo is free and does not require an API key.
"""

import os
import json
from datetime import datetime
from typing import Any
import requests

class WeatherMCPServer:
    """
    MCP server for weather data integration.
    Provides real-time weather information to improve demand and price predictions.
    """
    
    def __init__(self):
        self.weather_api_url = os.getenv("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
        self.geocoding_api_url = os.getenv("GEOCODING_API_URL", "https://geocoding-api.open-meteo.com/v1/search")
        
    def get_coordinates(self, location: str) -> dict:
        """
        Get latitude and longitude for a location using OpenMeteo Geocoding API
        """
        try:
            params = {
                "name": location,
                "count": 1,
                "language": "en",
                "format": "json"
            }
            response = requests.get(self.geocoding_api_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "latitude": result["latitude"],
                    "longitude": result["longitude"],
                    "name": result["name"],
                    "country": result.get("country", ""),
                    "timezone": result.get("timezone", "")
                }
            return None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None
    
    def get_current_weather(self, latitude: float, longitude: float) -> dict:
        """
        Get current weather conditions
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m",
                "temperature_unit": "celsius",
                "wind_speed_unit": "kmh",
                "precipitation_unit": "mm",
                "timezone": "auto"
            }
            response = requests.get(self.weather_api_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return {
                "location": {
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                    "timezone": data.get("timezone")
                },
                "current": data.get("current", {}),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Weather API error: {e}")
            return None
    
    def get_weather_forecast(self, latitude: float, longitude: float, days: int = 7) -> dict:
        """
        Get weather forecast for specified number of days
        Useful for supply chain planning and demand forecasting
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "auto",
                "forecast_days": min(days, 16)  # Open-Meteo max is 16 days
            }
            response = requests.get(self.weather_api_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            return {
                "location": {
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                    "timezone": data.get("timezone")
                },
                "daily": data.get("daily", {}),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Forecast API error: {e}")
            return None
    
    def get_weather_impact_factor(self, latitude: float, longitude: float) -> dict:
        """
        Calculate supply chain impact factor based on weather conditions
        Returns a factor that can influence pricing and demand predictions
        """
        weather = self.get_current_weather(latitude, longitude)
        if not weather:
            return {"impact_factor": 1.0, "warning": None}
        
        current = weather.get("current", {})
        temp = current.get("temperature_2m", 15)
        precip = current.get("precipitation", 0)
        wind_speed = current.get("wind_speed_10m", 0)
        
        impact_factor = 1.0
        warnings = []
        
        # Temperature impact
        if temp < 0:
            impact_factor *= 1.15  # Cold weather - logistics delays
            warnings.append("Below freezing - potential logistics delays")
        elif temp > 35:
            impact_factor *= 1.10  # Hot weather - cooling costs
            warnings.append("High temperature - increased cooling costs")
        
        # Precipitation impact
        if precip > 10:
            impact_factor *= 1.12
            warnings.append("Heavy precipitation - possible transport disruptions")
        
        # Wind impact
        if wind_speed > 30:
            impact_factor *= 1.08
            warnings.append("Strong winds - potential shipment delays")
        
        return {
            "impact_factor": round(impact_factor, 3),
            "temperature": temp,
            "precipitation": precip,
            "wind_speed": wind_speed,
            "warnings": warnings if warnings else None
        }


# Singleton instance
_weather_mcp = None

def get_weather_mcp() -> WeatherMCPServer:
    """Get or create Weather MCP server instance"""
    global _weather_mcp
    if _weather_mcp is None:
        _weather_mcp = WeatherMCPServer()
    return _weather_mcp


if __name__ == "__main__":
    # Example usage
    mcp = WeatherMCPServer()
    
    # Get coordinates for a location
    coords = mcp.get_coordinates("New York")
    print(f"New York coordinates: {coords}")
    
    if coords:
        # Get current weather
        weather = mcp.get_current_weather(coords["latitude"], coords["longitude"])
        print(f"Current weather: {json.dumps(weather, indent=2)}")
        
        # Get forecast
        forecast = mcp.get_weather_forecast(coords["latitude"], coords["longitude"])
        print(f"Forecast: {json.dumps(forecast, indent=2)}")
        
        # Get impact factor
        impact = mcp.get_weather_impact_factor(coords["latitude"], coords["longitude"])
        print(f"Weather impact: {json.dumps(impact, indent=2)}")
