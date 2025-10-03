# app.py — FastAPI (Kemetia)
from __future__ import annotations

import os, re, json, base64, tempfile, unicodedata, time, math
from functools import lru_cache
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# ------------------------- Config & clients -------------------------
load_dotenv()

SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")

AUDIO_WORKER_URL     = os.getenv("AUDIO_WORKER_URL", "https://kemetia-audio-worker.onrender.com").rstrip("/")
USE_AUDIO_EMB        = os.getenv("USE_AUDIO_EMB", "1") == "1"

# Seuils (ajustables dans Render)
TEXT_SIM_THRESHOLD   = float(os.getenv("TEXT_SIM_THRESHOLD", "0.80"))  # on durcit le texte
AUDIO_SIM_THRESHOLD  = float(os.getenv("AUDIO_SIM_THRESHOLD", "0.40"))  # on assouplit l’audio
HARD_AUDIO_ONLY      = os.getenv("HARD_AUDIO_ONLY", "1") == "1"        # 1 = on ignore le texte


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

app = FastAPI(title="Kemetia Backend")

# CORS permissif
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- CORS preflight explicit (en plus du CORSMiddleware) ---
from fastapi.responses import Response

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

# ------------------------- Utils texte -------------------------
def nk(s: str) -> str:
    t = (s or "").lower()
    t = unicodedata.normalize("NFD", t)
    t = re.sub(r"[\u0300-\u036f]", "", t)
    t = t.replace("ɔ", "o").replace("Ɔ", "o").replace("ɛ", "e").replace("Ɛ", "e")
    t = t.replace("ɖ", "d").replace("ŋ", "ng").replace("ƒ", "f")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def soft_canon(s: str) -> str:
    t = nk(s)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@app.get("/health_plus")
def health_plus():
    return {
        "ok": True,
        "has_openai": bool(openai_client is not None),
        "has_supabase": bool(supabase is not None),
        "audio_worker": AUDIO_WORKER_URL or None,
        "TEXT_SIM_THRESHOLD": TEXT_SIM_THRESHOLD,
        "AUDIO_SIM_THRESHOLD": AUDIO_SIM_THRESHOLD,
        "AUDIO_FORCE": AUDIO_FORCE,
    }

@app.get("/warmup")
def warmup():
    return {"ok": True}

# ------------------------- OpenAI Embeddings (texte) -------------------------
EMB_MODEL = "text-embedding-3-small"
EMBED_FAIL_UNTIL = 0.0

@lru_cache(maxsize=2048)
def _cached_embedding(q: str):
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=q)
    vec = resp.data[0].embedding
    return tuple(vec)

def embed_text(text: str) -> Optional[List[float]]:
    global EMBED_FAIL_UNTIL
    if not openai_client:
        return None
    now = time.time()
    if now < EMBED_FAIL_UNTIL:
        return None
    q = soft_canon(text)
    if not q:
        return None
    try:
        vec_tuple = _cached_embedding(q)
        return list(vec_tuple)
    except Exception as e:
        s = str(e) or ""
        if "RateLimit" in s or "quota" in s.lower() or "429" in s:
            EMBED_FAIL_UNTIL = time.time() + 120
        else:
            EMBED_FAIL_UNTIL = time.time() + 30
        print("❌ embed error (backoff set):", repr(e))
        return None

# ------------------------- Audio embedding via Worker -------------------------
def _embedding_via_worker(data_url: str) -> List[float]:
    if not AUDIO_WORKER_URL:
        print("⚠️ AUDIO_WORKER_URL manquant — embedding audio désactivé.")
        return []
    try:
        r = requests.post(
            f"{AUDIO_WORKER_URL}/api/compute_audio_embedding",
            json={"audio": data_url},
            timeout=60
        )
        r.raise_for_status()
        j = r.json()
        vec = j.get("embedding") or []
        return vec if isinstance(vec, list) else []
    except requests.HTTPError as he:
        print("audio-worker HTTP error:", getattr(he.response, "status_code", "??"), getattr(he.response, "text", "")[:200])
        return []
    except Exception as e:
        print("audio-worker exception:", e)
        return []

# ------------------------- Schemas -------------------------
class ChatIn(BaseModel):
    text: str = ""
    mode: str = "exchange"     # "exchange" | "translate"
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
    kind: str = "pharmacy"  # pharmacy | health | food
    lat: float
    lon: float
    radius: int = 4000

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
    """
    Réveille le worker + précharge OpenL3 avec un son silencieux.
    """
    try:
        # petit WAV silencieux base64 (16k mono ~200ms)
        import struct, base64
        sr = 16000
        samples = int(0.2 * sr)
        # header WAV
        header = b"RIFF" + struct.pack("<I", 36 + samples*2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr*2, 2, 16) + b"data" + struct.pack("<I", samples*2)
        data = b"\x00" * (samples*2)
        wav = header + data
        b64 = base64.b64encode(wav).decode("ascii")
        data_url = f"data:audio/wav;base64,{b64}"

        # Proxy compute → worker (ça réveille l’instance)
        _ = _embedding_via_worker(data_url)

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

class DebugAudioIn(BaseModel):
    audio: str
    lang: str = "mina"
    limit: int = 5

@app.post("/api/debug_audio_match")
def api_debug_audio_match(inp: DebugAudioIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    if not (isinstance(inp.audio, str) and inp.audio.startswith("data:")):
        raise HTTPException(status_code=400, detail="audio dataURL requis")

    avec = _embedding_via_worker(inp.audio)
    if not avec:
        raise HTTPException(status_code=500, detail="embedding vide (worker)")

    try:
        rpc = supabase.rpc("match_audio_by_vector", {
            "p_lang": inp.lang.lower(),
            "p_query": avec,
            "p_limit": max(1, min(inp.limit, 20))
        }).execute()
        rows = rpc.data or []
        # ajoute sim = 1 - distance
        for r in rows:
            r["sim"] = 1.0 - float(r.get("distance", 1.0))
        return {"candidates": rows[:inp.limit]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rpc error: {e}")

@app.get("/api/warmup")
def api_warmup():
    try:
        requests.get(f"{AUDIO_WORKER_URL}/health", timeout=10)
    except Exception:
        pass
    # micro-embedding (silence) pour réveiller TF/OpenL3
    try:
        silent = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
        _ = _embedding_via_worker(silent)
    except Exception:
        pass
    return {"ok": True}

# ------------------------- Chat -------------------------


@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    # --- Params ---
    user_text = (inp.text or "").strip()
    src_lang  = (inp.sourceLang or "mina").lower()
    tgt_lang  = (inp.targetLang or "fr").lower()

    base_lang = src_lang if src_lang != "fr" else tgt_lang
    if base_lang not in {"mina","bm","ee","ha","sw","fr","en"}:
        base_lang = "mina"

    best_text = None
    best_audio = None
    audio_candidates_dbg = None

    # --- Mode radical: AUDIO ONLY ---
    if HARD_AUDIO_ONLY:
        user_text = ""  # on neutralise le texte même si présent

    # 1) (Optionnel) TEXTE → embedding → match (désactivé si HARD_AUDIO_ONLY=1)
    if user_text and openai_client and not HARD_AUDIO_ONLY:
        try:
            qvec = embed_text(user_text)
            if qvec:
                rpc = supabase.rpc("match_audio_meta", {
                    "p_lang": base_lang,
                    "p_query": qvec,
                    "p_limit": 5
                }).execute()
                cand = rpc.data or []
                if cand:
                    b = cand[0]
                    sim = 1.0 - float(b.get("distance", 1.0))
                    best_text = {"row": b, "score": sim, "via": "embed"}
        except Exception as e:
            print("❌ rpc match (text) error:", e)

    # 2) AUDIO → worker → match (PRIORITAIRE)
    if (inp.from_audio or HARD_AUDIO_ONLY) and inp.audio:
        try:
            avec = _embedding_via_worker(inp.audio)
            if avec and supabase:
                rpc2 = supabase.rpc("match_audio_by_vector", {
                    "p_lang": base_lang,
                    "p_query": avec,
                    "p_limit": 5
                }).execute()
                cand2 = rpc2.data or []
                if cand2:
                    b2   = cand2[0]
                    sim2 = 1.0 - float(b2.get("distance", 1.0))
                    best_audio = {"row": b2, "score": sim2, "via": "audio-embed"}
                    # debug
                    audio_candidates_dbg = [{
                        "row_id": it.get("id"),
                        "sim": round(1.0 - float(it.get("distance", 1.0)), 4),
                        "text": (it.get("text") or "").strip(),
                        "fr": (it.get("fr") or "").strip(),
                        "reply_same": (it.get("reply_same") or "").strip(),
                    } for it in cand2]
        except Exception as e:
            print("❌ audio-embed via worker error:", e)

    # 3) Arbitrage — AUDIO > TEXTE (et seuils)
    final_hit, via = None, "fallback"

    # a) si on a de l’audio et score >= seuil → ON PREND
    if best_audio and best_audio["score"] >= AUDIO_SIM_THRESHOLD:
        final_hit = {"row": best_audio["row"], "score": best_audio["score"]}
        via = "audio-embed"
    # b) si on a de l’audio mais score < seuil… en mode dur on PREND QUAND MÊME (pour débug & casser la régression)
    elif best_audio and HARD_AUDIO_ONLY:
        final_hit = {"row": best_audio["row"], "score": best_audio["score"]}
        via = "audio-embed-forced"
    # c) sinon on tente le texte (seulement si non HARD_AUDIO_ONLY)
    elif best_text and not HARD_AUDIO_ONLY and best_text["score"] >= TEXT_SIM_THRESHOLD:
        final_hit = {"row": best_text["row"], "score": best_text["score"]}
        via = "embed"

    # 4) Réponse
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
                "best_text": None if not best_text else {
                    "row_id": best_text["row"].get("id"),
                    "score": round(best_text["score"], 4)
                },
                "best_audio": None if not best_audio else {
                    "row_id": best_audio["row"].get("id"),
                    "score": round(best_audio["score"], 4)
                },
                "note": "no match → default"
            }
        try:
            supabase.table("server_logs").insert({
                "kind": "no_match",
                "base_lang": base_lang,
                "meta": {"from_audio": bool(inp.from_audio), "audio_only": HARD_AUDIO_ONLY},
                "debug": json.dumps(out.get("debug", {}))
            }).execute()
        except Exception:
            pass
        return out

    r = final_hit["row"]
    rs, fr, en, tx = (r.get("reply_same") or "").strip(), (r.get("fr") or "").strip(), (r.get("en") or "").strip(), (r.get("text") or "").strip()

    # mapping réponse (inchangé)
    if inp.mode == "exchange" or tgt_lang == "same":
        reply = rs or tx or fr or en or ""
    else:
        if src_lang == "fr":
            if tgt_lang in {"mina","bm","ee","ha","sw"}:
                reply = tx or rs or en or fr or ""
            elif tgt_lang == "en":
                reply = en or fr or tx or rs or ""
            else:
                reply = fr or ""
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
            "best_text": None if not best_text else {
                "row_id": best_text["row"].get("id"),
                "score": round(best_text["score"],4)
            },
            "best_audio": None if not best_audio else {
                "row_id": best_audio["row"].get("id"),
                "score": round(best_audio["score"],4)
            },
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
    if vec:
        row["embedding"] = vec

    res = supabase.table("audio_meta").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert échoué")
    return {"ok": True, "id": res.data[0]["id"]}

# ------------------------- Learn -------------------------
@app.post("/api/learn")
def api_learn(inp: LearnIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    row = {
        "row_id": inp.row_id,
        "accepted": bool(inp.accepted),
        "input_type": inp.input_type,
        "correction_text": (inp.correction_text or None),
        "correction_row_id": inp.correction_row_id,
        "meta": {"ip": None, "ua": "fastapi"}
    }
    res = supabase.table("events").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert events échoué")
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
        header, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio invalide")

    mime = "application/octet-stream"
    try:
        m = header.split(";")[0]
        mime = m.split(":", 1)[1] or mime
    except Exception:
        pass

    ext = ".bin"
    if "webm" in mime: ext = ".webm"
    elif "ogg" in mime: ext = ".ogg"
    elif "mp4" in mime or "m4a" in mime: ext = ".m4a"
    elif "wav" in mime: ext = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(raw)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as fh:
            tr = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=fh
            )
        text = (getattr(tr, "text", "") or "").strip()
        return {"text": text}
    except Exception as e:
        print("❌ STT error:", repr(e))
        raise HTTPException(status_code=500, detail="STT error")
    finally:
        try: os.remove(tmp_path)
        except: pass

# ------------------------- Nearby via OSM -------------------------
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _osm_query_for(kind: str) -> str:
    if kind == "pharmacy":
        return '(node["amenity"="pharmacy"](around:{rad},{lat},{lon}););'
    if kind == "health":
        return '(' \
               'node["amenity"="hospital"](around:{rad},{lat},{lon});' \
               'node["amenity"="clinic"](around:{rad},{lat},{lon});' \
               'node["amenity"="doctors"](around:{rad},{lat},{lon});' \
               ');'
    # food
    return '(' \
           'node["amenity"="restaurant"](around:{rad},{lat},{lon});' \
           'node["amenity"="cafe"](around:{rad},{lat},{lon});' \
           'node["amenity"="fast_food"](around:{rad},{lat},{lon});' \
           'node["amenity"="food_court"](around:{rad},{lat},{lon});' \
           ');'

def _osm_nearby(kind: str, lat: float, lon: float, radius: int) -> List[dict]:
    # Overpass
    q_body = f"""
    [out:json][timeout:25];
    {_osm_query_for(kind).format(lat=lat, lon=lon, rad=radius)}
    out body;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=q_body.encode("utf-8"), timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("OSM error:", e)
        return []

    out = []
    for el in data.get("elements", []):
        if el.get("type") != "node": 
            continue
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or "(sans nom)"
        addr_parts = [tags.get(k) for k in ("addr:street","addr:housenumber","addr:city") if tags.get(k)]
        addr = ", ".join(addr_parts) if addr_parts else None
        lat2, lon2 = float(el["lat"]), float(el["lon"])
        dist = int(_haversine(lat, lon, lat2, lon2))
        cat = kind
        if tags.get("amenity"):
            cat = tags["amenity"]
        out.append({
            "name": name,
            "category": cat,
            "addr": addr,
            "lat": lat2,
            "lon": lon2,
            "distance_m": dist,
            "opening_hours": tags.get("opening_hours"),
            "osm_id": f'{el.get("type","node")}/{el.get("id")}'
        })
    out.sort(key=lambda x: x["distance_m"])
    return out

@app.post("/api/nearby")
def api_nearby(inp: NearbyIn):
    kind = (inp.kind or "pharmacy").lower()
    if kind not in {"pharmacy","health","food"}:
        kind = "pharmacy"
    items = _osm_nearby(kind, float(inp.lat), float(inp.lon), int(inp.radius or 4000))
    if not items:
        return {"items": [], "notice": "Aucun résultat dans ce rayon."}
    return {"items": items[:25]}

# ------------------------- Audio-embedding util (proxy vers worker) -------------------------
@app.post("/api/compute_audio_embedding")
def api_compute_audio_embedding(payload: dict):
    if not USE_AUDIO_EMB:
        raise HTTPException(status_code=501, detail="audio-embedding désactivé (set USE_AUDIO_EMB=1)")
    data_url = (payload or {}).get("audio") or ""
    if not (isinstance(data_url, str) and data_url.startswith("data:")):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    vec = _embedding_via_worker(data_url)
    if not vec:
        raise HTTPException(status_code=500, detail="embedding vide")
    return {"embedding": vec, "dim": len(vec)}
