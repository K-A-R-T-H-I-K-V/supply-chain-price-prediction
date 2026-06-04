import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# 1. LOAD ENV VARS BEFORE ANYTHING ELSE
from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 2. Import your clean services and existing MCPs
try:
    from mcp.weather_server import get_weather_mcp
    # Assuming market_context is moved to services, or adjust import if it is still in root
    from services.market_context import build_market_context_prompt 
    from services.suggester import generate_ai_suggestions, groq_client
except ImportError as e:
    print(f"⚠️ Import error during setup: {e}")
    # Setup fallbacks so the app doesn't crash if files are missing
    get_weather_mcp = None
    build_market_context_prompt = None
    generate_ai_suggestions = None
    groq_client = None

# --- Supabase admin client (server-side) ---
try:
    from supabase import create_client as create_supabase_client
except Exception:
    create_supabase_client = None

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('UPABASE_SERVICE_ROLE_KEY')
supabase_admin = None
if create_supabase_client and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase_admin = create_supabase_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✅ Supabase admin client initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize Supabase admin client: {e}")

# Initialize FastAPI App
app = FastAPI(title="Supply Chain Price Prediction API")

if not groq_client:
    print("⚠️ GROQ_API_KEY not set or suggester not loaded. Expert suggestions will use fallback mode.")

# Allow CORS so your React frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = Path(__file__).parent / "ml" / "model.pkl"
HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "prediction_history.json"
model = None
prediction_history: List[Dict[str, Any]] = []

# Initialize Weather MCP
weather_mcp = get_weather_mcp() if get_weather_mcp else None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Run train_model.py first.")
except Exception as e:
    print(f"❌ Error loading model: {e}")


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class PredictionRequest(BaseModel):
    order_quantity: int
    discount: float
    shipping_cost: float
    product_base_margin: float
    product_category: str
    month: int
    ship_mode: str
    order_priority: str

class InsightRequest(BaseModel):
    order_quantity: int
    discount: float
    shipping_cost: float
    product_base_margin: float
    product_category: str
    order_priority: Optional[str] = None
    ship_mode: Optional[str] = None

class ExpertSuggestionRequest(BaseModel):
    order_quantity: int
    discount: float
    shipping_cost: float
    product_base_margin: float
    product_category: str
    month: int
    ship_mode: str
    order_priority: str
    predicted_price: float

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_prediction_history() -> List[Dict[str, Any]]:
    if HISTORY_PATH.exists():
        try:
            with HISTORY_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []

def save_prediction_history() -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("w", encoding="utf-8") as f:
            json.dump(prediction_history, f, indent=2)
    except Exception as e:
        print(f"Failed to save prediction history: {e}")

def make_insight_recommendations(req: InsightRequest) -> List[str]:
    insights: List[str] = []

    if req.discount >= 0.15:
        insights.append("High discount levels are being used; monitor profitability closely.")
    elif req.discount >= 0.08:
        insights.append("Moderate discounts may improve volume while preserving margins.")
    else:
        insights.append("Low discounts preserve margin but may reduce order volume.")

    if req.order_quantity >= 40:
        insights.append("Large order quantity suggests bulk procurement and opportunity for carrier negotiation.")
    elif req.order_quantity >= 20:
        insights.append("Mid-sized orders are efficient for inventory turnover without too much storage cost.")
    else:
        insights.append("Small orders are more expensive to ship per unit; consider consolidation.")

    if req.shipping_cost > 40:
        insights.append("Shipping costs are high; review your logistics plan or alternative carriers.")
    else:
        insights.append("Shipping cost looks healthy for this order size.")

    if req.product_base_margin < 0.45:
        insights.append("Base margin is slim; prioritize higher-margin items or reduce discount exposure.")
    else:
        insights.append("Base margin is healthy and gives you pricing flexibility.")

    if req.product_category.lower() == "technology":
        insights.append("Technology products are price-sensitive; keep lead times short and stock levels optimized.")

    return insights

def make_chat_response(message: str) -> str:
    text = message.lower()

    if "weather" in text:
        return "Use /api/weather with a location or coordinates to see how weather may affect supply chain timing."
    if "discount" in text or "margin" in text:
        return "Focus on discount strategy only when it helps volume without eroding your gross margin too much."
    if "shipping" in text or "logistics" in text:
        return "Review shipping modes before purchase. If shipping is more than 10% of unit cost, choose a lower-cost mode."
    if "predict" in text or "price" in text:
        return "Send your order details to /api/predict for a unit price forecast."

    return "I can help with pricing, shipping, and insight recommendations. Ask about weather impact, discount strategy, or unit price prediction."

def make_fallback_suggestions(req: ExpertSuggestionRequest) -> Dict[str, Any]:
    """Fallback to rule-based suggestions when Groq is unavailable."""
    suggestions = []

    if req.predicted_price > 100:
        suggestions.append("Your predicted unit price is above typical levels — review supplier options and negotiate volume discounts.")
    else:
        suggestions.append("The estimated price is competitive; ensure fulfillment timelines are optimized to protect margins.")

    if req.discount > 0.08:
        suggestions.append("Consider tightening discounts if margin pressure grows, especially in technology and furniture categories.")
    else:
        suggestions.append("Discounts are conservative; focus on maintaining stock levels and reliable delivery flows.")

    if req.shipping_cost > 40:
        suggestions.append("Shipping is a meaningful cost driver. Explore alternative carriers or consolidate smaller shipments.")
    else:
        suggestions.append("Shipping cost is manageable; keep an eye on mode selection during peak demand months.")

    if req.product_base_margin < 0.45:
        suggestions.append("Margins are thin. Minimize discounting and avoid expensive expedited shipping unless absolutely needed.")
    else:
        suggestions.append("Margin headroom is healthy. Use it strategically for faster-moving categories or premium service levels.")

    if req.order_priority.lower() == "critical":
        suggestions.append("Critical orders should be monitored closely for delivery delays and expedited if needed.")

    if req.product_category.lower() == "technology":
        suggestions.append("Technology products benefit from tight inventory control due to fast depreciation and seasonality.")

    return {
        "success": True,
        "message": "",
        "suggestions": suggestions,
        "model": "fallback"
    }

def make_expert_suggestions(req: ExpertSuggestionRequest) -> Dict[str, Any]:
    """Generate expert suggestions using the dedicated suggester service."""
    context_data = {
        "order_quantity": req.order_quantity,
        "discount": f"{req.discount * 100:.1f}%",
        "shipping_cost": f"${req.shipping_cost:.2f}",
        "product_base_margin": f"{req.product_base_margin * 100:.1f}%",
        "product_category": req.product_category,
        "month": req.month,
        "ship_mode": req.ship_mode,
        "order_priority": req.order_priority,
        "predicted_price": f"${req.predicted_price:.2f}",
    }
    
    # Get current market context (weather + news + finance)
    market_context = ""
    if build_market_context_prompt:
        try:
            market_context = build_market_context_prompt(req.product_category, req.month)
        except Exception as e:
            print(f"Failed to fetch market context: {e}")
    
    # If Groq is available via our service, use it
    if groq_client and generate_ai_suggestions:
        result = generate_ai_suggestions(context_data, market_context)
        # If the LLM successfully generated a response, return it
        if "error" not in result:
            return result
            
    # Fallback to rule-based suggestions if anything fails
    return make_fallback_suggestions(req)


def get_summary_stats() -> Dict[str, Any]:
    total = len(prediction_history)
    average_price = round(sum(item["predicted_unit_price"] for item in prediction_history) / total, 2) if total else 0.0
    last_prediction = prediction_history[-1] if prediction_history else None
    return {
        "total_predictions": total,
        "average_predicted_unit_price": average_price,
        "last_prediction": last_prediction,
    }

prediction_history = load_prediction_history()


# -----------------------------
# Supabase token helpers
# -----------------------------
def get_user_from_token(authorization: Optional[str]):
    if not authorization:
        return None
    if not supabase_admin:
        return None
    token = authorization.split("Bearer ")[-1].strip()
    try:
        res = supabase_admin.auth.get_user(token)
        # supabase client returns a dict-like response; guard different shapes
        user = None
        if hasattr(res, 'user'):
            user = res.user
        elif isinstance(res, dict) and res.get('data') and res['data'].get('user'):
            user = res['data']['user']
        return user
    except Exception as e:
        print(f"Supabase token validation failed: {e}")
        return None


# ==========================================
# FASTAPI ENDPOINTS
# ==========================================
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/predict")
async def predict_price(req: PredictionRequest, authorization: Optional[str] = Header(None)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    try:
        quarter = (req.month - 1) // 3 + 1
        is_holiday = 1 if req.month in [11, 12] else 0
        shipping_cost_log = np.log1p(req.shipping_cost)

        if req.discount <= 0.05:
            discount_bin = 'low'
        elif req.discount <= 0.1:
            discount_bin = 'medium'
        else:
            discount_bin = 'high'
            
        discount_category = f"{discount_bin}_{req.product_category}"

        input_data = pd.DataFrame([{
            'Order Priority': req.order_priority,
            'Ship Mode': req.ship_mode,
            'Product Category': req.product_category,
            'Discount_Category': discount_category,
            'Order Quantity': req.order_quantity,
            'Discount': req.discount,
            'Shipping Cost Log': shipping_cost_log,
            'Product Base Margin': req.product_base_margin,
            'Month': req.month,
            'Quarter': quarter,
            'IsHoliday': is_holiday
        }])

        pred_log = model.predict(input_data)[0]
        pred_original = np.expm1(pred_log)
        predicted_price = round(float(pred_original), 2)

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request": req.dict(),
            "predicted_unit_price": predicted_price,
        }

        # Primary: Save to Supabase if available
        supabase_saved = False
        if supabase_admin:
            try:
                user = get_user_from_token(authorization)
                user_id = getattr(user, 'id', None) if user else None
                if not user_id:
                    user_id = str(uuid.uuid4())

                payload = {
                    'user_id': user_id,
                    'request': req.dict(),
                    'predicted_unit_price': predicted_price,
                    'expert_suggestions': None,
                    'created_at': entry['timestamp'],
                }
                supabase_admin.table('predictions').insert(payload).execute()
                print(f"✅ Prediction saved to Supabase for user: {user_id}")
                supabase_saved = True
            except Exception as e:
                print(f"⚠️ Failed to insert into Supabase: {e}")

        # Fallback: Save to local JSON only if Supabase failed
        if not supabase_saved:
            prediction_history.append(entry)
            save_prediction_history()
            print(f"📝 Prediction saved to local history (Supabase unavailable)")

        return {
            "success": True,
            "predicted_unit_price": predicted_price,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/api/weather")
async def weather(
    location: Optional[str] = Query(None, description="Location name, e.g. 'New York'"),
    latitude: Optional[float] = Query(None, description="Latitude coordinate"),
    longitude: Optional[float] = Query(None, description="Longitude coordinate"),
):
    if not weather_mcp:
        raise HTTPException(status_code=503, detail="Weather service unavailable.")

    if not location and (latitude is None or longitude is None):
        raise HTTPException(status_code=400, detail="Provide either location or both latitude and longitude.")

    if location:
        coords = weather_mcp.get_coordinates(location)
        if not coords:
            raise HTTPException(status_code=502, detail="Unable to resolve location.")
    else:
        coords = {
            "latitude": latitude,
            "longitude": longitude,
            "name": None,
            "country": None,
            "timezone": None,
        }

    weather_current = weather_mcp.get_current_weather(coords["latitude"], coords["longitude"])
    weather_forecast = weather_mcp.get_weather_forecast(coords["latitude"], coords["longitude"])
    weather_impact = weather_mcp.get_weather_impact_factor(coords["latitude"], coords["longitude"])

    if not weather_current:
        raise HTTPException(status_code=502, detail="Unable to fetch weather data.")

    return {
        "success": True,
        "location": coords,
        "current": weather_current.get("current"),
        "forecast": weather_forecast.get("daily") if weather_forecast else None,
        "impact": weather_impact,
    }

@app.post("/api/insights")
async def insights(req: InsightRequest):
    recommendations = make_insight_recommendations(req)
    return {
        "success": True,
        "summary": " ".join(recommendations),
        "insights": recommendations,
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    return {
        "success": True,
        "reply": make_chat_response(req.message),
    }

@app.post("/api/expert-suggestions")
async def expert_suggestions(req: ExpertSuggestionRequest):
    return make_expert_suggestions(req)


@app.post("/api/predictions")
async def save_prediction(req: PredictionRequest, authorization: Optional[str] = Header(None)):
    """Save a prediction to Supabase `predictions` table if available, otherwise keep local JSON file."""
    predicted_price = None
    try:
        # Recreate prediction locally first
        quarter = (req.month - 1) // 3 + 1
        shipping_cost_log = np.log1p(req.shipping_cost)
        # Use model if available
        if model:
            input_data = pd.DataFrame([{
                'Order Priority': req.order_priority,
                'Ship Mode': req.ship_mode,
                'Product Category': req.product_category,
                'Discount_Category': f"{req.discount}_{req.product_category}",
                'Order Quantity': req.order_quantity,
                'Discount': req.discount,
                'Shipping Cost Log': shipping_cost_log,
                'Product Base Margin': req.product_base_margin,
                'Month': req.month,
                'Quarter': quarter,
            }])
            pred_log = model.predict(input_data)[0]
            pred_original = np.expm1(pred_log)
            predicted_price = round(float(pred_original), 2)
        else:
            predicted_price = 0.0

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request": req.dict(),
            "predicted_unit_price": predicted_price,
        }

        # Persist to Supabase (with or without authentication)
        supabase_saved = False
        if supabase_admin:
            try:
                user = get_user_from_token(authorization)
                user_id = getattr(user, 'id', None) if user else None
                if not user_id:
                    user_id = str(uuid.uuid4())

                payload = {
                    'user_id': user_id,
                    'request': req.dict(),
                    'predicted_unit_price': predicted_price,
                    'expert_suggestions': None,
                }
                supabase_admin.table('predictions').insert(payload).execute()
                print(f"✅ Prediction saved to Supabase for user: {user_id}")
                supabase_saved = True
            except Exception as e:
                print(f"⚠️ Failed to insert into Supabase: {e}")

        # Fallback to local history if Supabase not available or failed
        if not supabase_saved:
            prediction_history.append(entry)
            save_prediction_history()
            print(f"📝 Prediction saved to local history (Supabase unavailable)")

        return {"success": True, "predicted_unit_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Save prediction error: {e}")


@app.get("/api/predictions/history")
async def get_prediction_history(authorization: Optional[str] = Header(None)):
    """Return prediction history from Supabase if available, otherwise return local history."""
    # Try to fetch from Supabase first (works for all users, both authenticated and anonymous)
    if supabase_admin:
        try:
            # Check if user is authenticated
            user = get_user_from_token(authorization)
            if user and getattr(user, 'id', None):
                # Authenticated user: return their specific predictions
                uid = getattr(user, 'id')
                resp = supabase_admin.table('predictions').select('id, user_id, request, predicted_unit_price, expert_suggestions, created_at')\
                    .eq('user_id', uid).order('created_at', {'ascending': False}).limit(50).execute()
            else:
                # Unauthenticated user: return all recent predictions
                resp = supabase_admin.table('predictions').select('id, user_id, request, predicted_unit_price, expert_suggestions, created_at')\
                    .order('created_at', {'ascending': False}).limit(50).execute()
            
            # Handle response format (may be dict with 'data' key or object with .data attribute)
            data = None
            if isinstance(resp, dict) and resp.get('data') is not None:
                data = resp['data']
            elif hasattr(resp, 'data'):
                data = resp.data
            
            print(f"✅ Fetched {len(data or [])} predictions from Supabase")
            return {"success": True, "history": data or [], "count": len(data or [])}
        except Exception as e:
            print(f"⚠️ Failed to fetch Supabase history: {e}")
            # Fallback to local history
            return {"success": True, "history": prediction_history, "count": len(prediction_history)}

    # Supabase not configured — return local
    return {"success": True, "history": prediction_history, "count": len(prediction_history)}

@app.delete("/api/predictions/history")
async def delete_prediction_history():
    prediction_history.clear()
    save_prediction_history()
    return {
        "success": True,
        "message": "Prediction history cleared.",
        "count": 0,
    }

@app.get("/api/stats")
async def stats():
    return {
        "success": True,
        "statistics": get_summary_stats(),
    }

# To run this server, use: uvicorn app:app --reload --port 5000