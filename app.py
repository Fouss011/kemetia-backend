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
import hashlib

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
TTS_VOICE = os.getenv("TTS_VOICE", "aria")  # voix féminine par défaut si dispo
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")

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
    country: str | None = None

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
    images_urls: Optional[List[str]] = None        # URLs déjà uploadées
    images_data: Optional[List[str]] = None        # dataURL (optionnel)

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
    "moudékoukou, né olédji hotel gblonbé moulé dji hotel, alo restaurant olédjia? gblonbé moulédji restaurant. nouké olédjia gbloinnam !"
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
# ------------------------- Debug audio-match -------------------------
@app.post("/api/debug_audio_match")
def api_debug_audio_match(inp: DebugAudioIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    if not (isinstance(inp.audio, str) and inp.audio.startswith("data:")):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    avec = _embedding_via_worker(inp.audio)
    if not avec:
        raise HTTPException(status_code=500, detail="embedding vide (worker)")
    rpc = supabase.rpc("match_audio_by_vector", {
        "p_lang": inp.lang.lower(),
        "p_query": avec,
        "p_limit": max(1, min(inp.limit, 20))
    }).execute()
    rows = rpc.data or []
    for r in rows:
        r["sim"] = 1.0 - float(r.get("distance", 1.0))
    return {"candidates": rows[:inp.limit]}

# ------------------------- Chat (règles Mina simplifiées) -------------------------
@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    user_text = (inp.text or "").strip()
    is_audio = bool(inp.from_audio or inp.audio)

    intent = None
    sim = None
    row_id = None

    # 1) PIPELINE AUDIO
    if is_audio and isinstance(inp.audio, str) and inp.audio.startswith("data:"):
        try:
            avec = _embedding_via_worker(inp.audio)
        except Exception as e:
            print("audio-worker err:", e)
            avec = []

        best = None
        if avec:
            try:
                rpc = supabase.rpc("match_audio_by_vector", {
                    "p_lang": (inp.sourceLang or "mina").lower(),
                    "p_query": avec,
                    "p_limit": 5
                }).execute()
                rows = rpc.data or []
                for r in rows:
                    r["sim"] = 1.0 - float(r.get("distance", 1.0))
                best = max(rows, key=lambda r: r.get("sim", 0.0), default=None)
            except Exception as e:
                print("RPC match_audio_by_vector err:", e)

        if best and best.get("sim", 0.0) >= AUDIO_SIM_THRESHOLD:
            sim = float(best["sim"])
            row_id = best.get("id")
            reply = (
                (best.get("reply_same") or "").strip()
                or (best.get("fr") or "").strip()
                or (best.get("en") or "").strip()
                or (best.get("text") or "").strip()
                or FALLBACK_MINA
            )
            # petit extract d’intent côté backend (fiable pour le front)
            it, _ = _match_intent_mina(reply)
            intent = it
            trigger = f"nearby:{'food' if intent=='restaurant' else intent}" if intent in ("pharmacy","restaurant") else None
            return {
                "reply": reply,
                "row_id": row_id,
                "mode": f"audio:match({sim:.2f})",
                "intent": intent,
                "sim": sim,
                "trigger": trigger
            }

        if HARD_AUDIO_ONLY:
            return {"reply": FALLBACK_MINA, "row_id": None, "mode": "audio:hard-fallback", "intent": None, "sim": None, "trigger": None}

        # STT fallback
        if openai_client:
            try:
                header, b64 = inp.audio.split(",", 1)
                raw = base64.b64decode(b64)
                mime = "application/octet-stream"
                try:
                    mime = header.split(";")[0].split(":", 1)[1] or mime
                except Exception:
                    pass
                ext = ".bin"
                if   "webm" in mime: ext = ".webm"
                elif "ogg"  in mime: ext = ".ogg"
                elif "mp4" in mime or "m4a" in mime: ext = ".m4a"
                elif "wav"  in mime: ext = ".wav"

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                    f.write(raw); tmp_path = f.name
                try:
                    with open(tmp_path, "rb") as fh:
                        tr = openai_client.audio.transcriptions.create(model="whisper-1", file=fh)
                    stt_text = (getattr(tr, "text", "") or "").strip()
                finally:
                    try: os.remove(tmp_path)
                    except: pass

                if stt_text:
                    user_text = stt_text
                else:
                    return {"reply": FALLBACK_MINA, "row_id": None, "mode": "audio:stt-empty", "intent": None, "sim": None, "trigger": None}
            except Exception as e:
                print("STT fallback err:", repr(e))
                return {"reply": FALLBACK_MINA, "row_id": None, "mode": "audio:stt-error", "intent": None, "sim": None, "trigger": None}
        else:
            return {"reply": FALLBACK_MINA, "row_id": None, "mode": "audio:no-openai", "intent": None, "sim": None, "trigger": None}

    # 2) INTRO
    if user_text == "" and not is_audio:
        return {"reply": INTRO_MINA, "row_id": None, "mode": "intro", "intent": None, "sim": None, "trigger": None}

    # 3) RÈGLES TEXTE
    intent, rule_reply = _match_intent_mina(user_text)
    if intent and rule_reply:
        trigger = f"nearby:{'food' if intent=='restaurant' else intent}" if intent in ("pharmacy","restaurant") else None
        return {"reply": rule_reply, "row_id": None, "mode": f"rule:{intent}", "intent": intent, "sim": None, "trigger": trigger}

    # 4) FALLBACK
    return {"reply": FALLBACK_MINA, "row_id": None, "mode": "fallback", "intent": None, "sim": None, "trigger": None}




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
    
@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload d'une image d'illustration d'évènement.
    Stocké dans le bucket public 'events-media' (dossier events/).
    Retourne l'URL publique.
    """
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier vide")

    ct = (file.content_type or "").lower()
    if not ct.startswith("image/"):
        raise HTTPException(status_code=400, detail="Content-Type image/* requis")

    ext = ".jpg"
    if "png" in ct:  ext = ".png"
    if "jpeg" in ct: ext = ".jpg"
    if "webp" in ct: ext = ".webp"

    key = f"events/{uuid4()}{ext}"
    BUCKET = "events-media"

    try:
        supabase.storage.from_(BUCKET).upload(
            key, data,
            file_options={"content-type": ct, "cache-control":"3600", "upsert":"false"}
        )
        url = supabase.storage.from_(BUCKET).get_public_url(key)
        if not url:
            raise RuntimeError("public URL vide")
        return {"url": url, "bucket": BUCKET, "key": key, "content_type": ct}
    except Exception as e:
        msg = str(e)
        if "Bucket not found" in msg or "No such bucket" in msg:
            raise HTTPException(400, "Bucket 'events-media' introuvable (créé-le en public).")
        raise HTTPException(500, f"Upload image fail: {msg}")


@app.post("/api/events_admin/delete")
def api_delete_event(req: EventId, authorization: Optional[str] = Header(None)):
    """Suppression d’un évènement par ID (admin)."""
    _require_admin(authorization)
    try:
        supabase.table("events").delete().eq("id", req.event_id).execute()
        return {"ok": True, "deleted_id": req.event_id}
    except Exception as e:
        raise HTTPException(500, f"delete fail: {e}")
    
# ---- ADMIN: lister publiés & refusés (avec filtre pays optionnel) ----
@app.get("/api/events_admin/list_all")
def api_events_admin_list_all(request: Request, country: str | None = None, horizon_days: int = 14, limit: int = 500):
    """
    Retourne:
      - published: événements publiés (table events), dans la fenêtre temporelle (par défaut 14j)
      - rejected:  propositions refusées (table event_submissions où accepted=false)
    Filtre optionnel par code pays (country_code), et horizon ajustable.
    """
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    token = auth.replace("Bearer", "").strip()
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # bornes temporelles pour published
    now = datetime.now(timezone.utc)
    horizon_days = max(1, min(int(horizon_days or 14), 90))  # 1..90 jours
    max_dt = now + timedelta(days=horizon_days)

    def parse_iso(s):
        if not s: return None
        try: return datetime.fromisoformat(str(s).replace("Z","+00:00"))
        except: return None

    # --- published (events)
    FIELDS = ",".join([
        "id","title","description","title_mina","description_mina",
        "venue_name","address","city","lat","lon","location_text",
        "starts_at:start_time","end_time","visibility","is_published","is_national",
        "price_text:price","price_amount","price_currency",
        "audio_url","cover_url","contact_phone","contact_url","country_code"
    ])

    try:
        q = supabase.table("events").select(FIELDS).eq("is_published", True).order("start_time", desc=False).limit(limit)
        res = q.execute(); rows = res.data or []
    except Exception as e:
        rows = []
        print("events_admin.list_all events error:", e)

    # filtre date + pays
    pub = []
    cc = (country or "").strip().upper()
    for r in rows:
        dt = r.get("starts_at")
        if not isinstance(dt, datetime):
            dt = parse_iso(dt)
        if not dt or not (now <= dt <= max_dt):
            continue
        if cc and (str(r.get("country_code") or "").upper() != cc):
            continue
        r["price_display"] = _fmt_price(r.get("price_amount"), r.get("price_currency"), r.get("price_text"))
        if isinstance(r.get("starts_at"), datetime):
            r["starts_at"] = r["starts_at"].isoformat()
        pub.append(r)

    # --- rejected (event_submissions)
    try:
        q2 = (supabase.table("event_submissions")
              .select("*")
              .eq("is_reviewed", True)
              .eq("is_approved", False)
              .order("created_at", desc=True)
              .limit(limit))
        rej = q2.execute().data or []
        if cc:
            rej = [x for x in rej if (str(x.get("country_code") or "").upper() == cc)]
    except Exception as e:
        rej = []
        print("events_admin.list_all submissions error:", e)

    return {"published": pub, "rejected": rej, "count_published": len(pub), "count_rejected": len(rej)}


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
        "country_code",
        "images",
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
            
        # ---- << COLLER ICI le filtre pays >> ----
        country = (q.country or "").strip().upper()
        if country:
    # TEMPORAIRE : inclure aussi les events sans code pays (NULL/"")
            normalized = [
                r for r in normalized
                if (str(r.get("country_code") or "").upper() == country) or (r.get("country_code") in (None, ""))
            ]
# ---- >> FIN DU FILTRE PAYS ----

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
# --- PRO: lecture publique de profils ---------------------------------------
from fastapi import Query

def _sb_headers(service: bool = False):
    """Récupère les headers Supabase (service si dispo, sinon anon)."""
    import os
    anon = getattr(SET, "SUPABASE_ANON_KEY", None) or os.getenv("SUPABASE_ANON_KEY")
    svc  = getattr(SET, "SUPABASE_SERVICE_KEY", None) or os.getenv("SUPABASE_SERVICE_KEY")
    key  = (svc if service else anon) or anon
    return {"apikey": key, "Authorization": f"Bearer {key}"}

def _sb_url(path: str) -> str:
    import os
    base = getattr(SET, "SUPABASE_URL", None) or os.getenv("SUPABASE_URL")
    return f"{base.rstrip('/')}{path}"

@app.get("/api/pro_public")
def api_pro_public(
    id: str | None = None,
    slug: str | None = None,
    sector: str | None = None,
    city: str | None = None,
    q: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    radius_km: float | None = None,
    limit: int = Query(500, ge=1, le=2000),
):
    """
    Liste les profils PRO.
    - Filtres: id, slug, sector, city, q (recherche nom/ville)
    - Option 'nearby' côté serveur (sans PostGIS): lat, lon, radius_km -> filtre sur colonnes lat/lon si présentes
    """
    import requests

    # 1) Construire la requête PostgREST
    url = _sb_url("/rest/v1/pro_profiles?select=id,slug,display_name,sector,city,phone,whatsapp,website,about,images,audio_url,lat,lon,created_at,updated_at&order=updated_at.desc&limit=" + str(limit))
    if id:
        url += f"&id=eq.{id}"
    if slug:
        url += f"&slug=eq.{slug}"
    if sector:
        url += f"&sector=eq.{sector}"
    if city:
        # recherche partielle sur city
        url += f"&city=ilike.*{city}*"
    if q:
        # recherche partielle sur display_name OU city (simple: on filtre city côté client ensuite)
        url += f"&display_name=ilike.*{q}*"

    r = requests.get(url, headers=_sb_headers(False), timeout=30)
    try:
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Supabase error: {getattr(e, 'response', r).text}")

    items = r.json() or []

    # 2) Filtre distance (si lat/lon fournis et radius_km défini)
    if lat is not None and lon is not None and radius_km is not None:
        from math import radians, sin, cos, asin, sqrt
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
            return 2 * R * asin(sqrt(a))

        filtered = []
        for it in items:
            la = it.get("lat"); lo = it.get("lon")
            if la is None or lo is None:
                continue
            try:
                d = haversine(float(lat), float(lon), float(la), float(lo))
            except Exception:
                continue
            if d <= float(radius_km):
                it["_dist_km"] = round(d, 2)
                filtered.append(it)
        filtered.sort(key=lambda x: x.get("_dist_km", 9e9))
        items = filtered

    return {"items": items}

# ------------------------- Submit Event -------------------------

      # ✅ URL déjà uploadée via /api/upload_audio


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
    
        # --- IMAGES (multi)
    images_urls: List[str] = []
    # a) URLs directes
    if inp.images_urls:
        for u in inp.images_urls:
            if isinstance(u, str) and u.startswith("http"):
                images_urls.append(u)

    # b) dataURL -> upload
    if inp.images_data:
        for durl in inp.images_data[:4]:  # limite 4 images
            try:
                head, b64 = durl.split(",", 1)
                raw = base64.b64decode(b64)
                mime = "image/jpeg"
                try: mime = head.split(";")[0].split(":",1)[1] or "image/jpeg"
                except: pass
                ext = ".jpg"
                if "png" in mime:  ext = ".png"
                if "webp" in mime: ext = ".webp"

                name = f"events/{uuid4()}{ext}"
                bucket = supabase.storage.from_("events-media")
                bucket.upload(name, raw, {"content-type": mime, "cache-control":"3600"})
                url = bucket.get_public_url(name)
                if url: images_urls.append(url)
            except Exception as e:
                print("image upload fail:", e)


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
        "audio_url": audio_url,
        "images": images_urls or [],    # ✅ on stocke l’URL (ou None)
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

    # --- Auth admin
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    token = auth.replace("Bearer", "").strip()
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # --- Récupérer la soumission
    r = supabase.table("event_submissions").select("*").eq("id", inp.submission_id).limit(1).execute()
    sub = (r.data or [None])[0]
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # --- Cas refus
    if not inp.accept:
        supabase.table("event_submissions").update({
            "accepted": False,
            "admin_note": inp.admin_note or "refused",
            "is_reviewed": True,
            "is_approved": False
        }).eq("id", inp.submission_id).execute()
        return {"ok": True, "published": False}

    # --- Visibilité
    vis = (sub.get("visibility") or "local").lower()
    if vis not in ("local","city","national","global"):
        vis = "local"

    # --- Fallback location_text propre
    location_text = " · ".join(
        x for x in [
            (sub.get("venue_name") or "").strip(),
            (sub.get("address") or "").strip(),
            (sub.get("city") or "").strip(),
        ] if x
    ) or (sub.get("venue_name") or sub.get("address") or sub.get("city") or None)

    # --- Construire la ligne pour 'events' (✅ cover_url + contacts + pays)
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
        "cover_url":        sub.get("cover_url"),        # ✅ image
        "contact_phone":    sub.get("contact_phone"),    # ✅ téléphone
        "contact_url":      sub.get("contact_url"),      # ✅ lien
        "country_code":     sub.get("country_code"),     # ✅ pays si présent
        "is_published":     True,
        "is_national":      True if vis in ("national","global") else False,
        "location_text":    location_text,
        "price_currency":   "XOF",
    }
    # Conserver explicitement les champs null autorisés
    pub = {k: v for k, v in pub.items() if v is not None or k in ("end_time","price")}

    # --- Insert dans events
    try:
        ins = supabase.table("events").insert(pub).execute()
        if not ins.data:
            return {"ok": False, "error": "insert events failed: no data"}
        new_id = ins.data[0]["id"]
    except Exception as e:
        return {"ok": False, "error": f"insert events fail: {e}"}

    # --- Marquer la soumission comme publiée
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
        model=TTS_MODEL, voice=TTS_VOICE, input=txt, format="mp3"
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

from functools import lru_cache
import hashlib

def _tts_cache_key(text: str, voice: str, model: str, rate: float) -> str:
    h = hashlib.sha1(f"{voice}|{model}|{rate:.2f}|{text}".encode("utf-8")).hexdigest()
    return h

@lru_cache(maxsize=512)
def _tts_b64_cached(text: str, voice: str, model: str, rate: float) -> str:
    # NB: certains TTS n’exposent pas "rate", on le laisse au modèle par défaut
    speech = openai_client.audio.speech.create(
        model=model, voice=voice, input=text, format="mp3"
    )
    mp3_bytes = getattr(speech, "content", None)
    if not mp3_bytes:
        # fallback streaming si nécessaire
        with openai_client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=text
        ) as resp:
            mp3_bytes = resp.read()
    return base64.b64encode(mp3_bytes).decode("ascii")

@app.post("/api/tts_chat")
def api_tts_chat(payload: dict):
    if not openai_client:
        raise HTTPException(status_code=501, detail="TTS non configuré (OPENAI_API_KEY manquant)")
    text = (payload or {}).get("text") or ""
    if not text.strip():
        raise HTTPException(status_code=400, detail="text requis")
    voice = (payload or {}).get("voice") or TTS_VOICE
    model = (payload or {}).get("model") or TTS_MODEL
    rate = float((payload or {}).get("rate") or 1.0)  # réservé si on étend

    try:
        b64 = _tts_b64_cached(text, voice, model, rate)
        return {
            "audio": f"data:audio/mpeg;base64,{b64}",
            "voice_used": voice,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

from fastapi import UploadFile, File, Form, HTTPException
from typing import Optional, List
import secrets, datetime, requests

@app.post("/api/pro_submit")
async def api_pro_submit(
    display_name: str = Form(...),
    slug: str = Form(...),
    sector: str = Form(...),
    city: str = Form(""),
    phone: str = Form(""),
    whatsapp: str = Form(""),
    website: str = Form(""),
    about: str = Form(""),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    edit_token: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    if supabase is None or not SET.SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase service non configuré")

    svc_headers = {
        "apikey": SET.SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SET.SUPABASE_SERVICE_KEY}",
    }

    # Upload helpers
    def put_storage(bucket: str, key: str, file: UploadFile):
        url = f"{SET.SUPABASE_URL}/storage/v1/object/{bucket}/{key}"
        data = file.file.read()
        r = requests.post(url, headers=svc_headers, data=data)
        if r.status_code not in (200, 201):
            raise HTTPException(status_code=400, detail=f"upload fail {bucket}/{key}: {r.text}")
        return f"{bucket}/{key}"

    now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base_folder = f"temetia-pro/{slug}-{now}-{secrets.token_hex(4)}"

    img_paths = []
    if images:
        imgs = images[:3]  # max 3
        for i, f in enumerate(imgs, start=1):
            ext = (f.filename or "img").split(".")[-1].lower()
            key = f"{base_folder}/img_{i}.{ext or 'jpg'}"
            img_paths.append( put_storage("temetia-pro-images", key, f) )

    audio_path = None
    if audio:
        ext = (audio.filename or "audio").split(".")[-1].lower()
        key = f"{base_folder}/intro.{ext or 'mp3'}"
        audio_path = put_storage("temetia-pro-audio", key, audio)

    # Generate or verify edit token
    token = edit_token or secrets.token_urlsafe(24)

    # Upsert pro_profiles
    rec = {
        "slug": slug,
        "display_name": display_name,
        "sector": sector,
        "city": city or None,
        "phone": phone or None,
        "whatsapp": whatsapp or None,
        "website": website or None,
        "about": about or None,            # nécessite colonne about (voir SQL ci-dessous)
        "lat": lat,
        "lon": lon,
        "images": img_paths if img_paths else None,
        "audio_url": audio_path,
        "edit_token": token,               # nécessite colonne edit_token
    }

    # Merge with existing if edit_token matches or slug exists + token matches
    # (2 colonnes suggérées: edit_token text unique, updated_at timestamp)
    # Exemple: utiliser PostgREST upsert
    url = f"{SET.SUPABASE_URL}/rest/v1/pro_profiles?slug=eq.{slug}"
    headers = {**svc_headers, "Content-Type":"application/json", "Prefer":"resolution=merge-duplicates"}
    import json
    r = requests.post(url, headers=headers, data=json.dumps(rec))
    if r.status_code not in (200,201,204):
        raise HTTPException(status_code=400, detail=f"upsert fail: {r.text}")

    return {"ok": True, "message":"Profil enregistré", "edit_token": token}


# --- KEMETIA: soumission / édition de profil PRO -----------------------------
from fastapi import UploadFile, File, Form, HTTPException
from typing import Optional, List
import os, secrets, datetime, json, requests

# ========== CONFIG ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Renommage Temetia -> Kemetia
BUCKET_IMG = "kemetia-pro-images"
BUCKET_AUDIO = "kemetia-pro-audio"
FOLDER_PREFIX = "kemetia-pro"

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("[WARN] Supabase service env vars missing; /api/pro_submit will fail")

# ========== HELPERS ==========
def _sb_headers(service: bool = False):
    key = SUPABASE_SERVICE_KEY if service else (SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY)
    return {"apikey": key, "Authorization": f"Bearer {key}"}

def _sb_url(path: str) -> str:
    return f"{SUPABASE_URL.rstrip('/')}{path}"

def _storage_public_url(bucket: str, key: str) -> str:
    # URL publique (si le bucket est PUBLIC dans Supabase Storage)
    return f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{key}"

def _upload_to_storage(bucket: str, key: str, uf: UploadFile) -> str:
    """
    Upload binaire dans Supabase Storage (POST /storage/v1/object/{bucket}/{key})
    Renvoie l'URL publique.
    """
    svc_headers = _sb_headers(service=True)
    url = _sb_url(f"/storage/v1/object/{bucket}/{key}")
    data = uf.file.read()
    r = requests.post(url, headers=svc_headers, data=data)
    if r.status_code not in (200, 201):
        raise HTTPException(status_code=400, detail=f"upload fail {bucket}/{key}: {r.text}")
    return _storage_public_url(bucket, key)

# ========== ROUTE ==========
@app.post("/api/pro_submit")
async def api_pro_submit(
    display_name: str = Form(...),
    slug: str = Form(...),
    sector: str = Form(...),
    city: str = Form(""),
    phone: str = Form(""),
    whatsapp: str = Form(""),
    website: str = Form(""),
    about: str = Form(""),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    edit_token: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None),  # max 3
    audio: Optional[UploadFile] = File(None),         # ≤ 30s (vérif côté serveur à ajouter si besoin)
):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    base_folder = f"{FOLDER_PREFIX}/{slug}-{now}-{secrets.token_hex(4)}"

    # --- IMAGES (max 3) ---
    img_urls: List[str] = []
    if images:
        imgs = images[:3]
        for i, f in enumerate(imgs, start=1):
            ext = (f.filename or "img").split(".")[-1].lower()
            if ext not in {"jpg", "jpeg", "png", "webp", "gif"}:
                ext = "jpg"
            key = f"{base_folder}/img_{i}.{ext}"
            img_urls.append(_upload_to_storage(BUCKET_IMG, key, f))

    # --- AUDIO (optionnel) ---
    audio_url = None
    if audio:
        ext = (audio.filename or "audio").split(".")[-1].lower()
        if ext not in {"mp3", "m4a", "aac", "wav", "ogg", "webm"}:
            ext = "mp3"
        key = f"{base_folder}/intro.{ext}"
        audio_url = _upload_to_storage(BUCKET_AUDIO, key, audio)

    # --- TOKEN D'ÉDITION ---
    token = edit_token or secrets.token_urlsafe(24)

    # --- UPSERT PRO_PROFILE (slug unique) ---
    rec = {
        "slug": slug,
        "display_name": display_name,
        "sector": sector,
        "city": city or None,
        "phone": phone or None,
        "whatsapp": whatsapp or None,
        "website": website or None,
        "about": about or None,
        "lat": lat,
        "lon": lon,
        "images": img_urls if img_urls else None,
        "audio_url": audio_url,
        "edit_token": token,
        "updated_at": datetime.datetime.utcnow().isoformat()
    }

    url = _sb_url(f"/rest/v1/pro_profiles?slug=eq.{slug}")
    headers = { **_sb_headers(service=True), "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates" }
    r = requests.post(url, headers=headers, data=json.dumps(rec))
    if r.status_code not in (200, 201, 204):
        raise HTTPException(status_code=400, detail=f"upsert fail: {r.text}")

    return {"ok": True, "message": "Profil enregistré", "edit_token": token}

@app.get("/api/pro_public")
def api_pro_public(q: str | None = None, sector: str | None = None, city: str | None = None, limit: int = 500):
    import requests, os
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    url = f"{SUPABASE_URL}/rest/v1/pro_profiles?select=id,slug,display_name,sector,city,phone,whatsapp,website,about,images,audio_url,lat,lon,created_at,updated_at&order=updated_at.desc&limit={limit}"
    if sector: url += f"&sector=eq.{sector}"
    if city:   url += f"&city=ilike.*{city}*"
    if q:      url += f"&display_name=ilike.*{q}*"
    r = requests.get(url, headers={"apikey":SUPABASE_ANON_KEY,"Authorization":f"Bearer {SUPABASE_ANON_KEY}"}, timeout=30)
    if r.status_code >= 400: raise HTTPException(status_code=502, detail=r.text)
    return {"items": r.json() or []}
