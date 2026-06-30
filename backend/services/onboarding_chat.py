import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# Plain-language copy — no tech terms, no sales pitch
OFFICE_ITEMS = "printer paper, pens, and staplers"
SCOPE_NOTE = (
    f"Right now we handle office supplies only — a small catalog with about "
    f"2–3 essentials like {OFFICE_ITEMS}."
)
LIVE_DATA_NOTE = (
    "Behind the scenes, the tool also looks at live signals — weather that "
    "could slow deliveries, supply-chain news, and fuel costs — so your "
    "forecast and tips reflect real conditions, not just spreadsheet math."
)
FORM_NOTE = (
    "You fill in a short form (order size, discount, shipping, and timing), "
    "and it returns a predicted unit price plus practical suggestions."
)

FALLBACK_EXPLAIN_YES = (
    "Good to know you're familiar with supply chain. "
    f"{SCOPE_NOTE} {FORM_NOTE} {LIVE_DATA_NOTE}"
)

FALLBACK_EXPLAIN_NO = (
    "No problem — supply chain simply means getting products from a supplier "
    "to your desk on time and at a fair price. "
    f"{SCOPE_NOTE} {FORM_NOTE} {LIVE_DATA_NOTE}"
)

INVITE_MESSAGE = (
    "Ready to try it? Click below and fill in the prediction form with your order details."
)

ONBOARDING_SYSTEM_PROMPT = """You explain a simple office-supplies price prediction tool to everyday users.

STRICT RULES — follow every one:
- Write 3–4 short sentences only. Plain, friendly tone — like talking to a coworker.
- Do NOT sound like a sales pitch. Ban words/phrases like: revolutionary, cutting-edge, leverage, synergy, game-changer, empower, seamless, robust, solution, platform, AI-powered (say "smart tips" or "helpful suggestions" instead).
- Do NOT mention any inner technology: no MCP, API, LLM, Groq, model, protocol, server, database, Supabase, JSON, or how systems are wired.
- DO say we focus on OFFICE SUPPLIES ONLY — a small catalog of about 2–3 items (e.g. printer paper, pens, staplers). Do not claim we cover furniture, tech, or broad retail.
- DO briefly explain that live outside information is woven in — weather that can delay shipments, relevant supply news, and fuel/market costs — so advice reflects real-world conditions. Describe this in everyday words, not as "integrations" or "data pipelines".
- DO mention the user fills a short form (quantity, discount, shipping, timing, etc.) and gets a price estimate plus practical tips.
- If the user said they do NOT know supply chain, add one plain sentence defining it (moving goods from supplier to customer on time).
- If the user said they DO know supply chain, skip the definition — just acknowledge briefly.
- No bullet points, no markdown, no headings. Under 110 words."""


def generate_onboarding_explanation(knows_supply_chain: bool) -> str:
    if not groq_client:
        return FALLBACK_EXPLAIN_YES if knows_supply_chain else FALLBACK_EXPLAIN_NO

    user_prompt = (
        f"The user {'already knows' if knows_supply_chain else 'does not know'} "
        "about supply chain. Write the onboarding reply."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ONBOARDING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=220,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or (FALLBACK_EXPLAIN_YES if knows_supply_chain else FALLBACK_EXPLAIN_NO)
    except Exception as e:
        print(f"Onboarding Groq error: {e}")
        return FALLBACK_EXPLAIN_YES if knows_supply_chain else FALLBACK_EXPLAIN_NO


def get_invite_message() -> str:
    return INVITE_MESSAGE
