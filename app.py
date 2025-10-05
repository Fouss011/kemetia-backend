# app.py — Kemetia (FastAPI, unifiée Oct 2025)
from __future__ import annotations
import os, re, json, base64, tempfile, unicodedata, time, math
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
from uuid import uuid4

# ------------------------- Config -------------------------
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
AUDIO_WORKER_URL     = os.getenv("AUDIO_WORKER_URL", "https://kemetia-audio-worker.onrender.com").rstrip("/")
USE_AUDIO_EMB        = os.getenv("USE_AUDIO_EMB", "1") == "1"
AUTO_PUBLISH = os.getenv("AUTO_PUBLISH", "0") == "1"

# Matching (ajustables via Render env)
TEXT_SIM_THRESHOLD   = float(os.getenv("TEXT_SIM_THRESHOLD", "0.80"))
AUDIO_SIM_THRESHOLD  = float(os.getenv("AUDIO_SIM_THRESHOLD", "0.40"))
HARD_AUDIO_ONLY      = os.getenv("HARD_AUDIO_ONLY", "1") == "1"  # “radical audio-first”

# Token admin pour modération d’évènements
ADMIN_TOKEN          = os.getenv("ADMIN_TOKEN", "")

# Clients
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        print("❌ Supabase init error:", e)

openai_client = None
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("❌ OpenAI init error:", e)

# App + CORS
app = FastAPI(title="Kemetia Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{full_path:path}", include_in_schema=False)
def any_options(full_path: str):
    resp = Response(status_code=204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    resp.headers["Access-Control-Max-Age"] = "600"
    return resp

@app.middleware("http")
async def ensure_cors_headers(request: Request, call_next):
    try:
        resp = await call_next(request)
    except Exception as e:
        from fastapi.responses import JSONResponse
        resp = JSONResponse({"detail": "server error", "error": str(e)}, status_code=500)
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
    resp.headers.setdefault("Access-Control-Max-Age", "600")
    return resp

# ------------------------- Utils texte & geo -------------------------
def nk(s: str) -> str:
    t = (s or "").lower()
    t = unicodedata.normalize("NFD", t)
    t = re.sub(r"[\u0300-\u036f]", "", t)
    t = t.replace("ɔ", "o").replace("Ɔ", "o").replace("ɛ", "e").replace("Ɛ", "e")
    t = t.replace("ɖ", "d").replace("ŋ", "ng").replace("ƒ", "f")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def soft_canon(s: str) -> str:
    t = nk(s); t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def canon_city(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return nk(s)

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ------------------------- Health & warmup -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "has_openai": bool(openai_client is not None),
        "has_supabase": bool(supabase is not None),
        "audio_worker": AUDIO_WORKER_URL or None
    }

@app.get("/warmup")
def warmup():
    """Réveille le worker (OpenL3) avec un mini WAV silencieux."""
    try:
        import struct
        sr = 16000; samples = int(0.2 * sr)
        header = b"RIFF" + struct.pack("<I", 36 + samples*2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16) + b"data" + struct.pack("<I", samples*2)
        data = b"\x00" * (samples*2)
        data_url = "data:audio/wav;base64," + base64.b64encode(header+data).decode("ascii")
        _ = _embedding_via_worker(data_url)
    except Exception as e:
        print("warmup err:", e)
    return {"ok": True}

# ------------------------- Embeddings -------------------------
EMB_MODEL = "text-embedding-3-small"
EMBED_FAIL_UNTIL = 0.0

@lru_cache(maxsize=2048)
def _cached_embedding(q: str):
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=q)
    return tuple(resp.data[0].embedding)

def embed_text(text: str) -> Optional[List[float]]:
    global EMBED_FAIL_UNTIL
    if not openai_client: return None
    if time.time() < EMBED_FAIL_UNTIL: return None
    q = soft_canon(text)
    if not q: return None
    try:
        return list(_cached_embedding(q))
    except Exception as e:
        s = str(e) or ""
        EMBED_FAIL_UNTIL = time.time() + (120 if ("429" in s or "quota" in s.lower()) else 30)
        print("❌ embed error:", s[:140])
        return None

def _embedding_via_worker(data_url: str) -> List[float]:
    if not AUDIO_WORKER_URL:
        print("⚠️ AUDIO_WORKER_URL manquant"); return []
    try:
        r = requests.post(f"{AUDIO_WORKER_URL}/api/compute_audio_embedding",
                          json={"audio": data_url}, timeout=60)
        r.raise_for_status()
        j = r.json()
        vec = j.get("embedding") or []
        return vec if isinstance(vec, list) else []
    except Exception as e:
        print("audio-worker error:", e); return []

# ------------------------- Schemas -------------------------
class ChatIn(BaseModel):
    text: str = ""
    mode: str = "exchange"
    sourceLang: str = "mina"
    targetLang: str = "fr"
    debug: bool = True
    bridge: bool = False
    from_audio: bool = False
    audio: Optional[str] = None
    history: Optional[List[Any]] = None

class CollectIn(BaseModel):
    lang: str
    category: Optional[str] = None
    text: str
    variants: Optional[str] = None
    reply_same: Optional[str] = None
    fr: Optional[str] = None
    en: Optional[str] = None
    filename: Optional[str] = None
    mime: Optional[str] = None
    audio: Optional[str] = None
    duration_ms: Optional[int] = None

class LearnIn(BaseModel):
    row_id: int
    accepted: bool
    input_type: str = "audio"
    correction_text: Optional[str] = None
    correction_row_id: Optional[int] = None

class NearbyIn(BaseModel):
    kind: str = "pharmacy"
    lat: float
    lon: float
    radius: int = 4000

class DebugAudioIn(BaseModel):
    audio: str
    lang: str = "mina"
    limit: int = 5

# Evénements
class EventsQuery(BaseModel):
    lat: float | None = None
    lon: float | None = None
    radius_m: int = 5000
    horizon_hours: int = 168   # 7 jours

class EventSubmitIn(BaseModel):
    title: str
    title_mina: Optional[str] = None
    description: Optional[str] = None
    description_mina: Optional[str] = None
    city: Optional[str] = None
    venue_name: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    start_time: str
    end_time: Optional[str] = None
    price: Optional[str] = None
    visibility: str = "local"  # 'local'|'city'|'national'|'global'
    contact_phone: Optional[str] = None
    contact_url: Optional[str] = None
    cover_url: Optional[str] = None
    # audio optionnel
    audio_data: Optional[str] = None     # dataURL (webm/ogg/mp3…)
    audio_duration_ms: Optional[int] = None
    audio_url: Optional[str] = None 

class EventPublishIn(BaseModel):
    submission_id: int
    accept: bool = True
    admin_note: Optional[str] = None

class EventId(BaseModel):
    event_id: int

# ====== RÈGLES MINA (intents simples) ======
INTRO_MINA = (
    "Woezon kaka ! Moudogbélo, oyonambé Vapayi ! "
    "Olé dji pharmacie, alo restaurantwo késolégbowoa, alo kondji ya, "
    "né olidji hotel né adon alon tchan, biyom pkoua ma soè dodawo ! Akpé !"
)

FALLBACK_MINA = (
    "moudékoukou gnémoukpolé wo édowanwo ! "
    "néolé dji pharmacie alo restaurant alo hotel, noudoudou tchan maté sowososawo apké!"
)

PHRASES_MINA = {
    "pharmacy": {
        "patterns": [
            "moulédji pharmacie",
            "ma plé atiké",
            "ma plé atchiké",
            "atiké",
            "moulé djila plé atiké",
            "pharmacie",
            "fikayé ma pko pharmacie léwo",
        ],
        "response": (
            "Yo, ma so pharmacie kéwo solé gbowoua dodawo fifidjin. "
            "Alo zi bouton ké yé djiyé atikékui léya ola pko pharmacie wo !"
        ),
    },
    "restaurant": {
        "patterns": [
            "adoléwoum",
            "adodéléwoumadé",
            "maplé nou ladou",
            "ma plé noudoudou",
            "moulédji restaurant wo",
            "restaurant",
            "ma plé éza",
            "éza",
            "noudoudou",
            "restaurant chic",
        ],
        "response": (
            "Yo, ma so restaurant kéwo solé gbowoua dodawo fifidjin. "
            "Alo zi bouton ké yé djiyé noudoudou léya ola pko noudoudou sapéwo !"
        ),
    },
    "hotel": {
        "patterns": [
            "alon lésom",
            "moulé djila don alon",
            "fikayé maté pko hotel léwo",
            "moulédji hotel",
            "moulédji motel",
            "hotel",
            "motel",
            "ma mlouagni",
        ],
        "response": (
            "Yo, ma so hotel alo motel kéwo solé gbowoua dodawo fifidjin. "
            "Alo zi bouton ké yé djiyé abati léya ola pko hotelwo késo légbowoa !"
        ),
    },
}

def _match_intent_mina(text: str):
    """Retourne ('pharmacy'|'restaurant'|'hotel', response) ou (None, None)."""
    q = nk(text)
    for intent, cfg in PHRASES_MINA.items():
        for pat in cfg["patterns"]:
            if nk(pat) in q or q in nk(pat):
                return intent, cfg["response"]
    return None, None

# ------------------------- Debug audio-match -------------------------
@app.post("/api/debug_audio_match")
def api_debug_audio_match(inp: DebugAudioIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    if not (isinstance(inp.audio, str) and inp.audio.startswith("data:")):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    avec = _embedding_via_worker(inp.audio)
    if not avec: raise HTTPException(status_code=500, detail="embedding vide (worker)")
    rpc = supabase.rpc("match_audio_by_vector", {
        "p_lang": inp.lang.lower(),
        "p_query": avec,
        "p_limit": max(1, min(inp.limit, 20))
    }).execute()
    rows = rpc.data or []
    for r in rows: r["sim"] = 1.0 - float(r.get("distance", 1.0))
    return {"candidates": rows[:inp.limit]}

# ------------------------- Chat (règles Mina simplifiées) -------------------------
@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    # 0) INTRO FORCÉE si aucun texte (appel initial du front)
    user_text = (inp.text or "").strip()
    if user_text == "" and not inp.from_audio:
        return {"reply": INTRO_MINA, "row_id": None, "mode": "intro"}

    # 1) INTENTS MINA (règles locales, prioritaire)
    intent, rule_reply = _match_intent_mina(user_text)
    if intent and rule_reply:
        return {"reply": rule_reply, "row_id": None, "mode": f"rule:{intent}"}

    # 2) FALLBACK MINA si non prévu
    return {"reply": FALLBACK_MINA, "row_id": None, "mode": "fallback"}

# ------------------------- Collect -------------------------
@app.post("/api/collect")
def api_collect(inp: CollectIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    row = {
        "lang": (inp.lang or "").lower(),
        "category": (inp.category or "").strip() or None,
        "text": (inp.text or "").strip(),
        "variants_text": (inp.variants or "").strip() or None,
        "reply_same": (inp.reply_same or "").strip() or None,
        "fr": (inp.fr or "").strip() or None,
        "en": (inp.en or "").strip() or None,
        "filename": (inp.filename or "").strip() or None,
        "mime": (inp.mime or "").strip() or None,
        "duration_ms": inp.duration_ms,
    }
    if not row["text"]:
        raise HTTPException(status_code=400, detail="text obligatoire")

    # Embedding texte
    vec = embed_text(row["text"])
    if vec: row["embedding"] = vec

    # Nouveau : audio -> upload + embedding
    audio_url = None
    if inp.audio:
        try:
            header, b64 = inp.audio.split(",", 1)
            raw = base64.b64decode(b64)
            mime = header.split(";")[0].split(":",1)[1]
            ext = ".webm"
            if "ogg" in mime: ext = ".ogg"
            elif "mp3" in mime or "mpeg" in mime: ext = ".mp3"
            elif "m4a" in mime or "mp4" in mime: ext = ".m4a"
            name = f"meta/{uuid4().hex}{ext}"

            bucket = supabase.storage.from_("kemetia-audio")  # bucket base audio
            bucket.upload(name, raw, {"content-type": mime, "cache-control": "3600"})
            audio_url = bucket.get_public_url(name)
            row["audio_url"] = audio_url

            # embedding audio via worker
            avec = _embedding_via_worker(inp.audio)
            if avec:
                row["audio_embedding"] = avec
        except Exception as e:
            print("collect audio fail:", e)

    res = supabase.table("audio_meta").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert échoué")
    return {"ok": True, "id": res.data[0]["id"], "audio_url": audio_url}

# ------------------------- Learn (feedback) -------------------------
@app.post("/api/learn")
def api_learn(inp: LearnIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    row = {
        "row_id": inp.row_id, "accepted": bool(inp.accepted),
        "input_type": inp.input_type, "correction_text": (inp.correction_text or None),
        "correction_row_id": inp.correction_row_id, "meta": {"ip": None, "ua": "fastapi"}
    }
    res = supabase.table("feedback").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert feedback échoué")
    return {"ok": True, "inserted": res.data[0]}

# ------------------------- STT (Whisper) -------------------------
@app.post("/api/stt")
def api_stt(payload: Dict[str, Any]):
    if not openai_client:
        raise HTTPException(status_code=501, detail="STT non configuré (OPENAI_API_KEY manquant)")
    data_url = (payload or {}).get("audio") or ""
    if not (isinstance(data_url, str) and data_url.startswith("data:")):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    try:
        header, b64 = data_url.split(",", 1); raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio invalide")
    mime = "application/octet-stream"
    try: mime = header.split(";")[0].split(":",1)[1] or mime
    except Exception: pass
    ext = ".bin"
    if   "webm" in mime: ext = ".webm"
    elif "ogg" in mime:  ext = ".ogg"
    elif "mp4" in mime or "m4a" in mime: ext = ".m4a"
    elif "wav" in mime:  ext = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(raw); tmp_path = f.name
    try:
        with open(tmp_path, "rb") as fh:
            tr = openai_client.audio.transcriptions.create(model="whisper-1", file=fh)
        return {"text": (getattr(tr, "text", "") or "").strip()}
    except Exception as e:
        print("❌ STT error:", repr(e)); raise HTTPException(status_code=500, detail="STT error")
    finally:
        try: os.remove(tmp_path)
        except: pass

# ------------------------- Admin helpers -------------------------
def _require_admin(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.replace("Bearer", "").strip()
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --- UPLOAD AUDIO (évènements) ---
@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio (multipart/form-data, champ 'file')
    et l’upload dans le bucket Supabase 'kemetia-audio' (dossier events/).
    Retourne une URL publique.
    """
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    # lecture du payload
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide")

    # extension à partir du content-type
    ct = (file.content_type or "").lower()
    ext = ".webm"
    if "ogg" in ct:      ext = ".ogg"
    elif "mp3" in ct or "mpeg" in ct: ext = ".mp3"
    elif "m4a" in ct or "mp4" in ct:  ext = ".m4a"
    elif "wav" in ct:    ext = ".wav"

    key = f"events/{uuid4()}{ext}"

    # nom du bucket utilisé PAR TOUT LE PROJET
    BUCKET = "kemetia-audio"   # ⚠️ crée-le dans Supabase storage si absent

    try:
        # vérifie l’existence du bucket en tentant un upload
        supabase.storage.from_(BUCKET).upload(
            key,
            data,
            file_options={
                "content-type": ct or "audio/webm",
                "cache-control": "3600",
                "upsert": "false",
            },
        )
        public = supabase.storage.from_(BUCKET).get_public_url(key)
        # sécurité: string attendu
        if not isinstance(public, str) or not public:
            raise RuntimeError("public URL vide")
        return {"url": public, "key": key, "bucket": BUCKET, "content_type": ct}
    except Exception as e:
        # aide au debug si le bucket n’existe pas
        msg = str(e)
        if "Bucket not found" in msg or "No such bucket" in msg:
            raise HTTPException(
                status_code=400,
                detail="Bucket not found: crée le bucket 'kemetia-audio' dans Supabase Storage (public)"
            )
        raise HTTPException(status_code=500, detail=f"Upload fail: {msg}")

@app.post("/api/events_admin/delete")
def api_delete_event(req: EventId, authorization: Optional[str] = Header(None)):
    """Suppression d’un évènement par ID (admin)."""
    _require_admin(authorization)
    try:
        supabase.table("events").delete().eq("id", req.event_id).execute()
        return {"ok": True, "deleted_id": req.event_id}
    except Exception as e:
        raise HTTPException(500, f"delete fail: {e}")

# ------------------------- Nearby (OSM) -------------------------
def _osm_query_for(kind: str) -> str:
    if kind == "pharmacy":
        return '(node["amenity"="pharmacy"](around:{rad},{lat},{lon}););'
    if kind == "health":
        return '(' \
               'node["amenity"="hospital"](around:{rad},{lat},{lon});' \
               'node["amenity"="clinic"](around:{rad},{lat},{lon});' \
               'node["amenity"="doctors"](around:{rad},{lat},{lon});' \
               ');'
    return '(' \
           'node["amenity"="restaurant"](around:{rad},{lat},{lon});' \
           'node["amenity"="cafe"](around:{rad},{lat},{lon});' \
           'node["amenity"="fast_food"](around:{rad},{lat},{lon});' \
           'node["amenity"="food_court"](around:{rad},{lat},{lon});' \
           ');'

def _osm_nearby(kind: str, lat: float, lon: float, radius: int) -> List[dict]:
    q_body = f"""[out:json][timeout:25];{_osm_query_for(kind).format(lat=lat, lon=lon, rad=radius)}out body;"""
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=q_body.encode("utf-8"), timeout=30)
        r.raise_for_status(); data = r.json()
    except Exception as e:
        print("OSM error:", e); return []
    out = []
    for el in data.get("elements", []):
        if el.get("type") != "node": continue
        tags = el.get("tags", {}) or {}; name = tags.get("name") or "(sans nom)"
        addr_parts = [tags.get(k) for k in ("addr:street","addr:housenumber","addr:city") if tags.get(k)]
        addr = ", ".join(addr_parts) if addr_parts else None
        lat2, lon2 = float(el["lat"]), float(el["lon"]); dist = int(_haversine(lat, lon, lat2, lon2))
        cat = tags.get("amenity") or kind
        out.append({"name": name, "category": cat, "addr": addr, "lat": lat2, "lon": lon2, "distance_m": dist})
    out.sort(key=lambda x: x["distance_m"]); return out

@app.post("/api/nearby")
def api_nearby(inp: NearbyIn):
    kind = (inp.kind or "pharmacy").lower()
    if kind not in {"pharmacy","health","food"}: kind = "pharmacy"
    items = _osm_nearby(kind, float(inp.lat), float(inp.lon), int(inp.radius or 4000))
    return {"items": items[:25]} if items else {"items": [], "notice": "Aucun résultat dans ce rayon."}

# ------------------------- Événements publics (API unifiée) -------------------------
def _fmt_price(amount, currency, text_fallback=None):
    """
    Affiche un prix soit au format numérique + devise, soit (fallback) le texte libre venant de la DB.
    """
    try:
        if amount is not None:
            a = float(amount)
            cur = (currency or "XOF").upper()
            return f"{int(a)} {cur}" if a.is_integer() else f"{a:.0f} {cur}"
    except Exception:
        pass
    return (text_fallback or None)

# ------------------------- Événements publics (API unifiée) -------------------------
@app.get("/api/events_public")
@app.post("/api/events_public")
def api_events_public(q: EventsQuery = EventsQuery(), request: Request = None):
    """
    Source de vérité = table 'events'.
    - On sélectionne les champs utiles avec alias (start_time → starts_at, price → price_text).
    - On filtre côté Python: dates (now..now+horizon) + géo (si lat/lon fournis).
    - Retour: { local: [...], national: [...], now: ISO }
    
    Query string ?debug=1 -> renvoie les erreurs en clair (utile en prod).
    """
    debug = (request.query_params.get("debug") == "1")

    if supabase is None:
        msg = "Supabase non configuré"
        if debug:
            return {"local": [], "national": [], "now": datetime.now(timezone.utc).isoformat(), "error": msg}
        raise HTTPException(status_code=500, detail=msg)

    # -- Fenêtre temporelle
    now = datetime.now(timezone.utc)
    try:
        horizon = int(q.horizon_hours or 168)
    except Exception:
        horizon = 168
    horizon = max(1, min(horizon, 24 * 14))  # 1h .. 14 jours
    max_dt = now + timedelta(hours=horizon)

    # -- Petits helpers locaux
    def parse_iso(s):
        if not s:
            return None
        try:
            # accepte 'Z' ou offset explicite
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:
            return None

    # -- Champs à lire (alias PostgREST)
    FIELDS = ",".join([
        "id",
        "title",
        "description",
        "title_mina",
        "description_mina",
        "venue_name",
        "address",
        "city",
        "lat",
        "lon",
        "location_text",
        "starts_at:start_time",        # ✅ alias standardisé
        "end_time",
        "visibility",
        "is_published",
        "is_national",
        "price_text:price",            # ✅ si tu stockes le prix libre en 'price'
        "price_amount",
        "price_currency",
        "audio_url",
        "cover_url",
        "contact_phone",
        "contact_url",
    ])

    # -- Lecture DB
    try:
        raw = (
            supabase.table("events")
            .select(FIELDS)
            .eq("is_published", True)          # on affiche seulement les publiés
            .order("start_time", desc=False)   # tri ascendant par date
            .limit(1000)
            .execute()
        ).data or []
    except Exception as e:
        if debug:
            return {"local": [], "national": [], "now": now.isoformat(), "error": f"DB read fail: {e!r}"}
        raise HTTPException(status_code=500, detail="DB read fail")

    # -- Normalisation / filtres
    try:
        normalized = []
        for r in raw:
            dt = r.get("starts_at")
            if isinstance(dt, datetime):
                pass
            else:
                dt = parse_iso(dt)

            # Filtre fenêtre temporelle
            if not dt or not (now <= dt <= max_dt):
                continue

            # Location text fallback (si vide)
            loc_txt = r.get("location_text")
            if not (isinstance(loc_txt, str) and loc_txt.strip()):
                loc_txt = " · ".join([x for x in [
                    (r.get("venue_name") or "").strip(),
                    (r.get("address") or "").strip(),
                    (r.get("city") or "").strip()
                ] if x]) or None

            # Prix affichable
            price_disp = _fmt_price(
                r.get("price_amount"),
                r.get("price_currency"),
                r.get("price_text")
            )

            # starts_at en ISO
            starts_at_iso = dt.isoformat()

            # Copie nettoyée
            it = dict(r)
            it["location_text"] = loc_txt
            it["price_display"] = price_disp
            it["starts_at"] = starts_at_iso
            normalized.append(it)

        # -- Séparation national / local
        # National = taggé national (ou global) -> on ordonne par date
        nat = [r for r in normalized if bool(r.get("is_national"))]
        nat.sort(key=lambda r: (r.get("starts_at") or ""))

        # Local = si coords fournies -> on filtre par rayon
        loc = []
        if q.lat is not None and q.lon is not None:
            blat, blon = float(q.lat), float(q.lon)
            try:
                rad = float(q.radius_m or 5000.0)
            except Exception:
                rad = 5000.0

            for r in normalized:
                try:
                    if r.get("lat") is None or r.get("lon") is None:
                        continue
                    d = _haversine(
                        blat, blon,
                        float(r["lat"]), float(r["lon"])
                    )
                except Exception:
                    continue
                if d <= rad:
                    r2 = dict(r)
                    r2["distance_m"] = int(d)
                    loc.append(r2)

            loc.sort(key=lambda r: (r.get("distance_m", 10**9), r.get("starts_at") or ""))

        return {
            "local": loc[:100],
            "national": nat[:100],
            "now": now.isoformat()
        }

    except Exception as e:
        if debug:
            return {
                "local": [],
                "national": [],
                "now": now.isoformat(),
                "error": f"postproc fail: {e!r}",
                "raw_count": len(raw)
            }
        raise HTTPException(status_code=500, detail="postproc fail")

@app.get("/api/events_probe")
def api_events_probe(limit: int = 5):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    try:
        rows = (
            supabase.table("events_api")
            .select("*")
            .order("starts_at", desc=True)  # montre les plus récents d’abord
            .limit(max(1, min(limit, 50)))
            .execute()
        ).data or []
        return {"items": rows, "count": len(rows)}
    except Exception as e:
        return {"items": [], "error": f"{e!r}"}

@app.get("/diag/versions")
def diag_versions():
    import pkg_resources
    def v(p):
        try: return pkg_resources.get_distribution(p).version
        except: return None
    return {
        "supabase": v("supabase"),
        "postgrest": v("postgrest"),
        "httpx": v("httpx")
    }

@app.get("/api/events/{event_id}")
def api_event_detail(event_id: int):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    r = supabase.table("events_api").select("*").eq("id", event_id).limit(1).execute()
    it = (r.data or [None])[0]
    if not it: raise HTTPException(status_code=404, detail="Event not found")
    it["price_display"] = _fmt_price(it.get("price_amount"), it.get("price_currency"), it.get("price_text"))
    if isinstance(it.get("starts_at"), datetime):
        it["starts_at"] = it["starts_at"].isoformat()
    return it

# ------------------------- Submit Event -------------------------
# ===================== Submit Event (complet) =====================

# ↳ Active le publish automatique si tu veux éviter l'admin
AUTO_PUBLISH = os.getenv("AUTO_PUBLISH", "0") == "1"

class EventSubmitIn(BaseModel):
    # Contenu principal
    title: str
    title_mina: Optional[str] = None
    description: Optional[str] = None
    description_mina: Optional[str] = None

    # Localisation
    city: Optional[str] = None
    venue_name: Optional[str] = None
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

    # Dates / prix / visibilité
    start_time: str
    end_time: Optional[str] = None
    price: Optional[str] = None
    visibility: str = "local"   # 'local' | 'city' | 'national' | 'global'

    # Contacts / média
    contact_phone: Optional[str] = None
    contact_url: Optional[str] = None
    cover_url: Optional[str] = None

    # Audio (2 voies)
    audio_data: Optional[str] = None         # dataURL (fallback: upload côté backend)
    audio_duration_ms: Optional[int] = None  # garde-fou (<= 30s)
    audio_url: Optional[str] = None          # ✅ URL déjà uploadée via /api/upload_audio


@app.post("/api/event_submit")
def api_event_submit(inp: EventSubmitIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    # --- Normalisation visibilité
    vis = (inp.visibility or "local").lower().strip()
    if vis not in ("local", "city", "national", "global"):
        vis = "local"

    # --- Champs obligatoires
    if not (inp.title or "").strip():
        raise HTTPException(status_code=400, detail="title requis")
    if not (inp.start_time or "").strip():
        raise HTTPException(status_code=400, detail="start_time requis")

    # --- Gestion Audio
    audio_url: Optional[str] = None

    if inp.audio_url:
        # 1) Cas simple : l’URL publique a déjà été obtenue via /api/upload_audio
        audio_url = (inp.audio_url or "").strip() or None

    elif inp.audio_data:
        # 2) Fallback : on reçoit un dataURL → on uploade dans le bucket 'events-audio'
        dur = int(inp.audio_duration_ms or 0)
        if dur <= 0 or dur > 31_000:
            raise HTTPException(status_code=400, detail="audio trop long (>30s)")

        try:
            header, b64 = inp.audio_data.split(",", 1)
            raw = base64.b64decode(b64)

            # Déduire l’extension/mime
            mime = "audio/webm"
            try:
                mime = header.split(";")[0].split(":", 1)[1] or "audio/webm"
            except Exception:
                pass

            ext = ".webm"
            hl = header.lower()
            if "audio/ogg" in hl:                   ext = ".ogg"
            elif "audio/mp3" in hl or "mpeg" in hl: ext = ".mp3"
            elif "audio/mp4" in hl or "m4a" in hl:  ext = ".m4a"
            elif "audio/wav" in hl:                 ext = ".wav"

            name = f"{uuid4().hex}{ext}"
            bucket = supabase.storage.from_("events-audio")
            bucket.upload(name, raw, {"content-type": mime, "cache-control": "3600"})
            audio_url = bucket.get_public_url(name)

            if not isinstance(audio_url, str) or not audio_url:
                raise RuntimeError("public URL vide après upload")

        except Exception as e:
            msg = str(e)
            if "Bucket not found" in msg or "No such bucket" in msg:
                raise HTTPException(
                    status_code=400,
                    detail="Bucket 'events-audio' introuvable. Crée-le dans Supabase Storage (public).",
                )
            raise HTTPException(status_code=500, detail=f"upload audio fail: {msg}")

    # --- Insertion de la proposition
    sub_row = {
        "title": (inp.title or "").strip(),
        "title_mina": (inp.title_mina or None),
        "description": (inp.description or None),
        "description_mina": (inp.description_mina or None),
        "city": (inp.city or None),
        "venue_name": (inp.venue_name or None),
        "address": (inp.address or None),
        "lat": inp.lat,
        "lon": inp.lon,
        "start_time": inp.start_time,
        "end_time": inp.end_time,
        "price": (inp.price or None),
        "visibility": vis,
        "contact_phone": (inp.contact_phone or None),
        "contact_url": (inp.contact_url or None),
        "cover_url": (inp.cover_url or None),
        "accepted": None,          # en attente
        "is_reviewed": False,
        "is_approved": False,
        "audio_url": audio_url,    # ✅ on stocke l’URL (ou None)
    }

    try:
        ins = supabase.table("event_submissions").insert(sub_row).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"insert submissions fail: {e}")

    if not ins.data:
        raise HTTPException(status_code=500, detail="Insert proposition échoué")

    submission = ins.data[0]
    out = {"ok": True, "submission_id": submission["id"]}

    # --- Optionnel : publish automatique (bypass admin)
    if AUTO_PUBLISH:
        try:
            # Calcule un location_text propre (fallback si pas défini par ailleurs)
            location_text = " · ".join(
                x for x in [
                    (submission.get("venue_name") or "").strip(),
                    (submission.get("address") or "").strip(),
                    (submission.get("city") or "").strip(),
                ]
                if x
            ) or None

            pub = {
                "title":            submission.get("title"),
                "title_mina":       submission.get("title_mina"),
                "description":      submission.get("description"),
                "description_mina": submission.get("description_mina"),
                "city":             submission.get("city"),
                "venue_name":       submission.get("venue_name"),
                "address":          submission.get("address"),
                "lat":              submission.get("lat"),
                "lon":              submission.get("lon"),
                "start_time":       submission.get("start_time"),
                "end_time":         submission.get("end_time"),
                "price":            submission.get("price"),
                "visibility":       vis,
                "audio_url":        submission.get("audio_url"),
                "is_published":     True,
                "is_national":      True if vis in ("national","global") else False,
                "location_text":    location_text,
                "price_currency":   "XOF",
            }
            # Garder des clés nulles explicites si besoin (end_time/price)
            pub = {k: v for k, v in pub.items() if v is not None or k in ("end_time", "price")}

            ev_ins = supabase.table("events").insert(pub).execute()
            if ev_ins.data:
                out["event_id"] = ev_ins.data[0]["id"]

            # marque la soumission comme acceptée
            supabase.table("event_submissions").update({
                "accepted": True,
                "is_reviewed": True,
                "is_approved": True,
                "admin_note": "auto-publish",
            }).eq("id", submission["id"]).execute()

            out["published"] = True

        except Exception as e:
            # on ne casse pas la réponse en cas d'échec d’auto-publish
            out["published"] = False
            out["warn"] = f"auto-publish fail: {e}"

    return out


@app.get("/api/submissions_probe")
def submissions_probe(limit: int = 5):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    try:
        rows = (supabase.table("event_submissions")
                .select("*")
                .order("created_at", desc=True)
                .limit(max(1, min(limit, 50)))
                .execute()).data or []
        return {"items": rows, "count": len(rows)}
    except Exception as e:
        return {"items": [], "error": f"{e!r}"}

@app.get("/api/submissions_columns")
def submissions_columns():
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    try:
        rows = (supabase.table("event_submissions")
                .select("*")
                .limit(1)
                .execute()).data or []
        cols = list(rows[0].keys()) if rows else None
        return {"columns": cols, "hint": "Si 'columns' est null, insère une ligne test pour révéler les noms."}
    except Exception as e:
        return {"columns": None, "error": f"{e!r}"}

# ---- ADMIN: lister les propositions en attente ----
@app.get("/api/events_admin/pending")
def api_events_admin_pending(request: Request, limit: int = 200):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    debug = (request.query_params.get("debug") == "1")

    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    token = auth.replace("Bearer", "").strip()
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        q = (supabase.table("event_submissions")
             .select("*")
             .order("created_at", desc=True)
             .limit(max(1, min(limit, 500))))
        q = q.is_("accepted", None)
        res = q.execute()
        return {"items": res.data or []}
    except Exception as e:
        if debug: return {"items": [], "error": f"{e!r}"}
        raise HTTPException(status_code=500, detail="DB error")

# ---- ADMIN: publier/refuser une proposition ----
@app.post("/api/events_admin/publish")
def api_events_admin_publish(inp: EventPublishIn, request: Request):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    token = auth.replace("Bearer", "").strip()
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    r = supabase.table("event_submissions").select("*").eq("id", inp.submission_id).limit(1).execute()
    sub = (r.data or [None])[0]
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Refus
    if not inp.accept:
        supabase.table("event_submissions").update({
            "accepted": False,
            "admin_note": inp.admin_note or "refused",
            "is_reviewed": True,
            "is_approved": False
        }).eq("id", inp.submission_id).execute()
        return {"ok": True, "published": False}

    # Fallback visibility
    vis = (sub.get("visibility") or "local").lower()
    if vis not in ("local","city","national","global"):
        vis = "local"

    pub = {
        "title":            sub.get("title"),
        "title_mina":       sub.get("title_mina"),
        "description":      sub.get("description"),
        "description_mina": sub.get("description_mina"),
        "city":             sub.get("city"),
        "venue_name":       sub.get("venue_name"),
        "address":          sub.get("address"),
        "lat":              sub.get("lat"),
        "lon":              sub.get("lon"),
        "start_time":       sub.get("start_time") or None,
        "end_time":         sub.get("end_time")   or None,
        "price":            sub.get("price"),
        "visibility":       vis,
        "audio_url":        sub.get("audio_url"),
        "is_published":     True,
        "is_national":      True if vis in ("national","global") else False,
        "location_text":    (sub.get("venue_name") or sub.get("address") or sub.get("city") or None),
        "price_currency":   "XOF",
        # 👉 contacts
        "contact_phone":    sub.get("contact_phone"),
        "contact_url":      sub.get("contact_url"),
    }
    pub = {k: v for k, v in pub.items() if v is not None or k in ("end_time","price")}

    try:
        ins = supabase.table("events").insert(pub).execute()
        if not ins.data:
            return {"ok": False, "error": "insert events failed: no data"}
        new_id = ins.data[0]["id"]
    except Exception as e:
        return {"ok": False, "error": f"insert events fail: {e}"}

    try:
        supabase.table("event_submissions").update({
            "accepted": True,
            "admin_note": inp.admin_note or "published",
            "is_reviewed": True,
            "is_approved": True
        }).eq("id", inp.submission_id).execute()
    except Exception as e:
        return {"ok": True, "published": True, "event_id": new_id, "warn": f"submission update fail: {e!r}"}

    return {"ok": True, "published": True, "event_id": new_id}

# ---------- Event short announce (TTS ~30s) ----------
def _announce_text(ev: dict) -> str:
    title = (ev.get("title") or "").strip()
    when  = (ev.get("starts_at") or ev.get("start_time") or "").strip()
    where = " · ".join([x for x in [(ev.get("venue_name") or "").strip(),
                                    (ev.get("location_text") or "").strip(),
                                    (ev.get("city") or "").strip()] if x])
    price = (ev.get("price_text") or ev.get("price_display") or "").strip()
    parts = [f"Événement : {title}."]
    if when:  parts.append(f"Date et heure : {when}.")
    if where: parts.append(f"Lieu : {where}.")
    if price: parts.append(f"Tarif : {price}.")
    parts.append("Ne manquez pas !")
    return " ".join(parts)

@lru_cache(maxsize=256)
def _tts_b64_for_event(event_id: int) -> str:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY manquant")
    r = supabase.table("events").select("*").eq("id", event_id).limit(1).execute()
    ev = (r.data or [None])[0]
    if not ev:
        raise RuntimeError("event not found")
    txt = _announce_text(ev)
    try:
        speech = openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=txt,
            format="mp3"
        )
        mp3_bytes = speech.content  # bytes
    except Exception:
        with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice="alloy", input=txt
        ) as resp:
            mp3_bytes = resp.read()
    return base64.b64encode(mp3_bytes).decode("ascii")

@app.get("/api/event_announce_preview")
def api_event_announce_preview(event_id: int):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    try:
        b64 = _tts_b64_for_event(int(event_id))
        return {"audio": f"data:audio/mpeg;base64,{b64}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

# ------------------------- Audio embedding proxy -------------------------
@app.post("/api/compute_audio_embedding")
def api_compute_audio_embedding(payload: dict):
    if not USE_AUDIO_EMB: raise HTTPException(status_code=501, detail="audio-embedding désactivé")
    data_url = (payload or {}).get("audio") or ""
    if not (isinstance(data_url, str) and data_url.startswith("data:")):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    vec = _embedding_via_worker(data_url)
    if not vec: raise HTTPException(status_code=500, detail="embedding vide")
    return {"embedding": vec, "dim": len(vec)}
