# app.py — Kemetia (FastAPI, unifiée Oct 2025)
from __future__ import annotations
import os, re, json, base64, tempfile, unicodedata, time, math
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# ------------------------- Config -------------------------
load_dotenv()
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
AUDIO_WORKER_URL     = os.getenv("AUDIO_WORKER_URL", "https://kemetia-audio-worker.onrender.com").rstrip("/")
USE_AUDIO_EMB        = os.getenv("USE_AUDIO_EMB", "1") == "1"

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

class EventPublishIn(BaseModel):
    submission_id: int
    accept: bool = True
    admin_note: Optional[str] = None

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

# ------------------------- Chat (audio-first) -------------------------
@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    user_text = (inp.text or "").strip()
    src_lang  = (inp.sourceLang or "mina").lower()
    tgt_lang  = (inp.targetLang or "fr").lower()
    base_lang = src_lang if src_lang != "fr" else tgt_lang
    if base_lang not in {"mina","bm","ee","ha","sw","fr","en"}: base_lang = "mina"

    best_text = None
    best_audio = None
    audio_candidates_dbg = None

    if HARD_AUDIO_ONLY:
        user_text = ""  # on neutralise le texte

    # 1) texte (si autorisé)
    if user_text and openai_client and not HARD_AUDIO_ONLY:
        try:
            qvec = embed_text(user_text)
            if qvec:
                rpc = supabase.rpc("match_audio_meta", {
                    "p_lang": base_lang, "p_query": qvec, "p_limit": 5
                }).execute()
                cand = rpc.data or []
                if cand:
                    b = cand[0]; sim = 1.0 - float(b.get("distance", 1.0))
                    best_text = {"row": b, "score": sim, "via": "embed"}
        except Exception as e:
            print("rpc match (text) error:", e)

    # 2) audio prioritaire
    if (inp.from_audio or HARD_AUDIO_ONLY) and inp.audio:
        try:
            avec = _embedding_via_worker(inp.audio)
            if avec and supabase:
                rpc2 = supabase.rpc("match_audio_by_vector", {
                    "p_lang": base_lang, "p_query": avec, "p_limit": 5
                }).execute()
                cand2 = rpc2.data or []
                if cand2:
                    b2   = cand2[0]; sim2 = 1.0 - float(b2.get("distance", 1.0))
                    best_audio = {"row": b2, "score": sim2, "via": "audio-embed"}
                    audio_candidates_dbg = [{
                        "row_id": it.get("id"),
                        "sim": round(1.0 - float(it.get("distance", 1.0)), 4),
                        "text": (it.get("text") or "").strip(),
                        "fr": (it.get("fr") or "").strip(),
                        "reply_same": (it.get("reply_same") or "").strip(),
                    } for it in cand2]
        except Exception as e:
            print("audio-embed error:", e)

    # 3) arbitrage
    final_hit, via = None, "fallback"
    if best_audio and best_audio["score"] >= AUDIO_SIM_THRESHOLD:
        final_hit = {"row": best_audio["row"], "score": best_audio["score"]}; via = "audio-embed"
    elif best_audio and HARD_AUDIO_ONLY:
        final_hit = {"row": best_audio["row"], "score": best_audio["score"]}; via = "audio-embed-forced"
    elif best_text and not HARD_AUDIO_ONLY and best_text["score"] >= TEXT_SIM_THRESHOLD:
        final_hit = {"row": best_text["row"], "score": best_text["score"]}; via = "embed"

    # 4) réponse
    if not final_hit:
        default_mina = "moudékoukou gnémousséwo"
        out = {"reply": default_mina, "row_id": None}
        if inp.debug:
            out["debug"] = {
                "from_audio": bool(inp.from_audio),
                "has_text": bool(user_text),
                "via": via,
                "mode_audio_only": HARD_AUDIO_ONLY,
                "AUDIO_SIM_THRESHOLD": AUDIO_SIM_THRESHOLD,
                "TEXT_SIM_THRESHOLD": TEXT_SIM_THRESHOLD,
                "baseLang": base_lang,
                "audio_candidates": audio_candidates_dbg,
                "best_text": None if not best_text else {"row_id": best_text["row"].get("id"), "score": round(best_text["score"],4)},
                "best_audio": None if not best_audio else {"row_id": best_audio["row"].get("id"), "score": round(best_audio["score"],4)},
                "note": "no match → default"
            }
        try:
            supabase.table("server_logs").insert({
                "kind": "no_match", "base_lang": base_lang,
                "meta": {"from_audio": bool(inp.from_audio), "audio_only": HARD_AUDIO_ONLY},
                "debug": json.dumps(out.get("debug", {}))
            }).execute()
        except Exception:
            pass
        return out

    r = final_hit["row"]
    rs, fr, en, tx = (r.get("reply_same") or "").strip(), (r.get("fr") or "").strip(), (r.get("en") or "").strip(), (r.get("text") or "").strip()

    if inp.mode == "exchange" or tgt_lang == "same":
        reply = rs or tx or fr or en or ""
    else:
        if src_lang == "fr":
            if tgt_lang in {"mina","bm","ee","ha","sw"}: reply = tx or rs or en or fr or ""
            elif tgt_lang == "en": reply = en or fr or tx or rs or ""
            else: reply = fr or ""
        else:
            if   tgt_lang == "fr": reply = fr or tx or rs or en or ""
            elif tgt_lang == "en": reply = en or fr or tx or rs or ""
            elif tgt_lang == src_lang: reply = tx or rs or fr or en or ""
            else: reply = tx or rs or fr or en or ""

    out = {"reply": reply, "row_id": r.get("id")}
    if inp.debug:
        out["debug"] = {
            "from_audio": bool(inp.from_audio),
            "has_text": bool(user_text),
            "via": via,
            "mode_audio_only": HARD_AUDIO_ONLY,
            "AUDIO_SIM_THRESHOLD": AUDIO_SIM_THRESHOLD,
            "TEXT_SIM_THRESHOLD": TEXT_SIM_THRESHOLD,
            "baseLang": base_lang,
            "best_text": None if not best_text else {"row_id": best_text["row"].get("id"), "score": round(best_text["score"],4)},
            "best_audio": None if not best_audio else {"row_id": best_audio["row"].get("id"), "score": round(best_audio["score"],4)},
            "audio_candidates": audio_candidates_dbg,
            "chosen_row_id": r.get("id")
        }
    return out

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
    vec = embed_text(row["text"])
    if vec: row["embedding"] = vec
    res = supabase.table("audio_meta").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert échoué")
    return {"ok": True, "id": res.data[0]["id"]}

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


@app.post("/api/events_public")
def api_events_public(q: EventsQuery, request: Request):
    """
    Lit la vue 'events_api' puis filtre en Python (dates + géo).
    Si debug=1 (query string), renvoie l'erreur en clair au lieu d'un 500.
    """
    debug = (request.query_params.get("debug") == "1")

    if supabase is None:
        msg = "Supabase non configuré"
        if debug: return {"local": [], "national": [], "now": datetime.now(timezone.utc).isoformat(), "error": msg}
        raise HTTPException(status_code=500, detail=msg)

    now = datetime.now(timezone.utc)
    horizon = max(1, min(q.horizon_hours or 168, 24*14))
    max_dt = now + timedelta(hours=horizon)

    FIELDS = ",".join([
        "id","title","description","location_text","lat","lon",
        "starts_at","is_national","price_amount","price_currency","price_text","audio_url"
    ])

    def parse_iso(s):
        if not s: return None
        try: return datetime.fromisoformat(str(s).replace("Z","+00:00"))
        except: return None

    try:
        # 1) lecture brute de la vue (pas de gte/lte)
        raw = (
            supabase.table("events_api")
            .select(FIELDS)
            .order("starts_at", desc=False)   # tri soft (même si text)
            .limit(1000)
            .execute()
        ).data or []
    except Exception as e:
        if debug: return {"local": [], "national": [], "now": now.isoformat(), "error": f"DB read fail: {e!r}"}
        raise HTTPException(status_code=500, detail="DB read fail")

    try:
        # 2) normalisation + fenêtre de temps
        normalized = []
        for r in raw:
            dt = parse_iso(r.get("starts_at"))
            if not dt or not (now <= dt <= max_dt):   # hors fenêtre
                continue
            # prix joli
            r["price_display"] = _fmt_price(r.get("price_amount"), r.get("price_currency"), r.get("price_text"))
            # ensure iso string
            if isinstance(r.get("starts_at"), datetime):
                r["starts_at"] = r["starts_at"].isoformat()
            normalized.append(r)

        # 3) split national / local
        nat = [r for r in normalized if bool(r.get("is_national"))]
        nat.sort(key=lambda r: (r.get("starts_at") or ""))

        loc = []
        if q.lat is not None and q.lon is not None:
            blat, blon = float(q.lat), float(q.lon)
            rad = float(q.radius_m or 5000)
            for r in normalized:
                try:
                    d = _haversine(blat, blon, float(r["lat"]), float(r["lon"]))
                except Exception:
                    continue
                if d <= rad:
                    r2 = dict(r); r2["distance_m"] = int(d); loc.append(r2)
            loc.sort(key=lambda r: (r.get("distance_m", 10**9), r.get("starts_at") or ""))

        return {"local": loc[:100], "national": nat[:100], "now": now.isoformat()}
    except Exception as e:
        if debug:
            return {"local": [], "national": [], "now": now.isoformat(), "error": f"postproc fail: {e!r}", "raw_count": len(raw)}
        raise HTTPException(status_code=500, detail="postproc fail")


@app.get("/api/events_probe")
def api_events_probe(limit: int = 5):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    try:
        rows = (
            supabase.table("events_api")
            .select("*")
            .order("starts_at", desc=False)
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
    # Normalise le prix pour l’affichage si besoin
    it["price_display"] = _fmt_price(it.get("price_amount"), it.get("price_currency"), it.get("price_text"))
    if isinstance(it.get("starts_at"), datetime):
        it["starts_at"] = it["starts_at"].isoformat()
    return it


@app.post("/api/event_submit")
def api_event_submit(inp: EventSubmitIn, request: Request):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    debug = (request.query_params.get("debug") == "1")

    vis = (inp.visibility or "local").lower()
    if vis not in ("local","city","national","global"):
        vis = "local"

    row = {
        "title": (inp.title or "").strip(),
        "title_mina": inp.title_mina,
        "description": inp.description,
        "description_mina": inp.description_mina,
        "city": inp.city,
        "venue_name": inp.venue_name,
        "address": inp.address,
        "lat": inp.lat,
        "lon": inp.lon,
        "start_time": inp.start_time,
        "end_time": inp.end_time,
        "price": inp.price,
        "visibility": vis,
        "contact_phone": inp.contact_phone,
        "contact_url": inp.contact_url,
        "cover_url": inp.cover_url
        # NE PAS mettre "accepted" ici → null par défaut
    }

    if not row["title"] or not row["start_time"]:
        msg = "title et start_time requis"
        if debug: return {"ok": False, "error": msg, "row": row}
        raise HTTPException(status_code=400, detail=msg)

    try:
        res = supabase.table("event_submissions").insert(row).execute()
        if not res.data:
            if debug: return {"ok": False, "error": "insert vide", "row": row}
            raise HTTPException(status_code=500, detail="Insert proposition échoué")
        return {"ok": True, "submission_id": res.data[0]["id"]}
    except Exception as e:
        if debug: return {"ok": False, "error": f"{e!r}", "row": row}
        raise HTTPException(status_code=500, detail="Insert proposition échoué")


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
    # Liste les colonnes vues par PostgREST (utile si erreur de schéma)
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    try:
        # PostgREST n’a pas d’endpoint “describe”, on tente une ligne vide
        rows = (supabase.table("event_submissions")
                .select("*")
                .limit(1)
                .execute()).data or []
        cols = list(rows[0].keys()) if rows else None
        return {"columns": cols, "hint": "Si 'columns' est null, insère une ligne test pour révéler les noms."}
    except Exception as e:
        return {"columns": None, "error": f"{e!r}"}

# ---- ADMIN: lister les propositions en attente (debug-friendly) ----
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
        # filtre accepted IS NULL si la colonne existe
        # (elle existe après le SQL ci-dessus)
        q = q.is_("accepted", None)
        res = q.execute()
        return {"items": res.data or []}
    except Exception as e:
        if debug: return {"items": [], "error": f"{e!r}"}
        raise HTTPException(status_code=500, detail="DB error")


# ---- ADMIN: publier/refuser une proposition (debug-friendly) ----
@app.post("/api/events_admin/publish")
def api_events_admin_publish(inp: EventPublishIn, request: Request):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    debug = (request.query_params.get("debug") == "1")

    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    token = auth.replace("Bearer", "").strip()
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        r = supabase.table("event_submissions").select("*").eq("id", inp.submission_id).limit(1).execute()
        sub = (r.data or [None])[0]
    except Exception as e:
        if debug: return {"ok": False, "error": f"read submissions fail: {e!r}"}
        raise HTTPException(status_code=500, detail="read submissions fail")

    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    if not inp.accept:
        try:
            supabase.table("event_submissions").update({"accepted": False, "admin_note": inp.admin_note or "refused"}).eq("id", inp.submission_id).execute()
            return {"ok": True, "published": False}
        except Exception as e:
            if debug: return {"ok": False, "error": f"update refuse fail: {e!r}"}
            raise HTTPException(status_code=500, detail="update refuse fail")

    try:
        pub = {k: sub.get(k) for k in [
            "title","title_mina","description","description_mina",
            "city","venue_name","address","lat","lon",
            "start_time","end_time","price","visibility",
            "contact_phone","contact_url","cover_url"
        ]}
        pub["is_published"] = True
        ins = supabase.table("events").insert(pub).execute()
        new_id = ins.data[0]["id"]
    except Exception as e:
        if debug: return {"ok": False, "error": f"insert events fail: {e!r}"}
        raise HTTPException(status_code=500, detail="insert events fail")

    try:
        supabase.table("event_submissions").update({"accepted": True, "admin_note": inp.admin_note or "published"}).eq("id", inp.submission_id).execute()
    except Exception as e:
        if debug: return {"ok": True, "published": True, "event_id": new_id, "warn": f"update submission note fail: {e!r}"}

    return {"ok": True, "published": True, "event_id": new_id}


# ---------- Event short announce (TTS ~30s) ----------
from functools import lru_cache

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
    # retourne base64 MP3 ou lève Exception
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY manquant")
    # récupérer l’event
    r = supabase.table("events").select("*").eq("id", event_id).limit(1).execute()
    ev = (r.data or [None])[0]
    if not ev:
        raise RuntimeError("event not found")
    txt = _announce_text(ev)
    # TTS OpenAI (MP3)
    try:
        # SDK v1
        speech = openai_client.audio.speech.create(
            model="gpt-4o-mini-tts",  # sinon "tts-1"
            voice="alloy",
            input=txt,
            format="mp3"
        )
        mp3_bytes = speech.content  # bytes
    except Exception:
        # fallback API older style (selon version SDK)
        with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts", voice="alloy", input=txt
        ) as resp:
            mp3_bytes = resp.read()
    import base64
    return base64.b64encode(mp3_bytes).decode("ascii")

@app.get("/api/event_announce_preview")
def api_event_announce_preview(event_id: int):
    """
    Renvoie un MP3 dataURL (30s max) pour annoncer l’événement.
    """
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
