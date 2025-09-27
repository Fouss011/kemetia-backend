import requests
from typing import List, Dict, Any
from .settings import SET

HEADERS = {
    "apikey": SET.SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SET.SUPABASE_SERVICE_KEY}",
    "accept": "application/json",
}

def get_audio_rows(limit: int = 5000, lang: str | None = None) -> List[Dict[str, Any]]:
    url = f"{SET.SUPABASE_URL}/rest/v1/audio_meta"
    params = {
        "select": "id,lang,category,text,reply_same,fr,en,variants_text,variants_sig,created_at",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    if lang:
        params["lang"] = f"eq.{lang}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_feedback_events(limit: int = 2000) -> List[Dict[str, Any]]:
    url = f"{SET.SUPABASE_URL}/rest/v1/events"
    params = {
        "select": "id,row_id,accepted,input_type,correction_text,created_at",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()
