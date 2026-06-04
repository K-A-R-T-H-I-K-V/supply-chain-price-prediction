# services/suggester.py
import os
import json
from groq import Groq
from dotenv import load_dotenv
from services.prompt import get_expert_system_prompt

# Load environment variables FIRST
load_dotenv()

# Initialize Groq Client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

if not groq_client:
    print("⚠️ GROQ_API_KEY not found in .env! Expert suggestions will fail or use fallback.")

def generate_ai_suggestions(req_data: dict, market_context: str) -> dict:
    if not groq_client:
        return {"error": "Groq client not initialized. Check API keys."}

    prompt = get_expert_system_prompt(req_data, market_context)

    try:
        # Using the current, active Llama 3.3 model for fast, accurate JSON generation
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # <--- Correct, active model ID
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Low temperature for more deterministic business advice
            response_format={"type": "json_object"} # Forces JSON output
        )
        
        raw_text = response.choices[0].message.content.strip()
        parsed_json = json.loads(raw_text)
        
        return {
            "success": True,
            "message": parsed_json.get("greeting", "") + "\n\n" + parsed_json.get("scenario", ""),
            "suggestions": format_suggestions(parsed_json.get("recommendations", [])),
            "summary": parsed_json.get("summary", ""),
            "model": "llama-3.3-70b-versatile", # <--- Updated this line to match!
            "context_used": "Real-time weather, news, and market data"
        }
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return {"error": str(e)}
    
def format_suggestions(recommendations: list) -> list:
    """Formats the JSON recommendations into UI-friendly strings"""
    formatted = []
    for rec in recommendations:
        title = rec.get("title", "")
        text = rec.get("text", "")
        tags = rec.get("source_tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        formatted.append(f"**{title}**: {text}{tag_str}")
    return formatted