# app.py — FastAPI (Texte via OpenAI + STT Whisper + audio-embedding via worker + Nearby OSM + FSM)
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

# Seuils (tu peux régler via env)
TEXT_SIM_THRESHOLD   = float(os.getenv("TEXT_SIM_THRESHOLD", "0.72"))
AUDIO_SIM_THRESHOLD  = float(os.getenv("AUDIO_SIM_THRESHOLD", "0.75"))

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

app = FastAPI(title="Kemetia Backend (light)")

# CORS permissif + ensure headers même en cas d’exception
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    history: Optional[Dict[str, Any]] = None  # prev_row_id, prev_expect

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

# ------------------------- Health -------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "has_openai": bool(openai_client is not None),
        "has_supabase": bool(supabase is not None),
        "audio_worker": AUDIO_WORKER_URL or None
    }

def log_server(kind: str, payload: dict):
    try:
        if supabase:
            supabase.table("server_logs").insert({"kind": kind, "debug": json.dumps(payload)}).execute()
    except Exception:
        pass

# ------------------------- Helpers FSM -------------------------
YES_WORDS = {
    "fr":   {"oui","ouais","daccord","ok","yes"},
    "mina": {"ee","ayi","éé","ayii","eyo","yo"},
    "en":   {"yes","yeah","yep","sure","ok"},
}
NO_WORDS = {
    "fr":   {"non","nope","nan"},
    "mina": {"ko","ao","ayi o","kpakpa"},
    "en":   {"no","nope","nah"},
}
MAYBE_WORDS = {
    "fr":   {"je sais pas","je ne sais pas","peut etre","peut-être","bof"},
    "mina": {"menyɔ","menyo","mande","manɖe"},
    "en":   {"maybe","not sure","idk"},
}
def classify_yn(user_text: str, lang: str) -> str|None:
    s = (user_text or "").lower()
    lang = (lang or "fr").lower()
    for w in YES_WORDS.get(lang, set()):
        if w in s: return "yes"
    for w in NO_WORDS.get(lang, set()):
        if w in s: return "no"
    for w in MAYBE_WORDS.get(lang, set()):
        if w in s: return "maybe"
    return None

def pick_option(user_text: str, options: dict) -> int|None:
    s = (user_text or "").lower()
    for k, v in (options or {}).items():
        if k and k.lower() in s:
            try: return int(v)
            except: pass
    return None

def pick_reply(src, tgt, tx, rs, fr, en, mode):
    if mode == "exchange" or tgt == "same":
        return rs or tx or fr or en or ""
    if tgt == "fr":   return fr or tx or rs or en or ""
    if tgt == "en":   return en or fr or tx or rs or ""
    if tgt in {"mina","bm"}:
        return tx or rs or fr or en or ""
    return tx or rs or fr or en or ""

# ------------------------- Chat -------------------------
@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    user_text = (inp.text or "").strip()
    # Base language côté ressources (priorise langues locales)
    BASE_DEFAULT = "mina"
    src = (inp.sourceLang or "mina").lower()
    tgt = (inp.targetLang or "fr").lower()
    base_lang = src if src in {"mina","bm","ee","ha","sw"} else (tgt if tgt in {"mina","bm","ee","ha","sw"} else BASE_DEFAULT)

    # ===== Flow branch si on répond à une question précédente
    hist = inp.history or {}
    prev_id = hist.get("prev_row_id")
    prev_expect = (hist.get("prev_expect") or "").lower().strip()
    if prev_id and prev_expect in {"yn","choice"} and user_text:
        try:
            prev = supabase.table("audio_meta").select(
                "id,lang,fsm_expect,fsm_yes_id,fsm_no_id,fsm_maybe_id,fsm_options,fsm_next_id,text,reply_same,fr,en"
            ).eq("id", prev_id).limit(1).execute().data
            prev = prev[0] if prev else None
        except Exception:
            prev = None

        if prev and (prev.get("fsm_expect") or "none") in {"yn","choice"}:
            nxt_id = None
            if prev["fsm_expect"] == "yn":
                yn = classify_yn(user_text, lang=prev.get("lang") or base_lang)
                if yn == "yes":   nxt_id = prev.get("fsm_yes_id")
                elif yn == "no":  nxt_id = prev.get("fsm_no_id")
                elif yn == "maybe": nxt_id = prev.get("fsm_maybe_id")
            else:  # choice
                nxt_id = pick_option(user_text, prev.get("fsm_options") or {}) or prev.get("fsm_next_id")

            if nxt_id:
                try:
                    nx = supabase.table("audio_meta").select("id,text,reply_same,fr,en,fsm_expect").eq("id", nxt_id).limit(1).execute().data
                    nx = nx[0] if nx else None
                except Exception:
                    nx = None
                if nx:
                    rs, fr, en, tx = (nx.get("reply_same") or "").strip(), (nx.get("fr") or "").strip(), (nx.get("en") or "").strip(), (nx.get("text") or "").strip()
                    reply = pick_reply(src, tgt, tx, rs, fr, en, inp.mode)
                    out = {"reply": reply, "row_id": nx.get("id")}
                    if inp.debug:
                        out["debug"] = {"from_audio": bool(inp.from_audio), "has_text": bool(user_text), "via":"flow", "prev_row_id": prev_id, "branch": prev.get("fsm_expect"), "chosen_row_id": nx.get("id"), "expect": (nx.get("fsm_expect") or "none")}
                    return out
            # Rien reconnu → répéter la question
            rs, fr, en, tx = (prev.get("reply_same") or "").strip(), (prev.get("fr") or "").strip(), (prev.get("en") or "").strip(), (prev.get("text") or "").strip()
            out = {"reply": pick_reply(src, tgt, tx, rs, fr, en, inp.mode), "row_id": prev.get("id")}
            if inp.debug:
                out["debug"] = {"from_audio": bool(inp.from_audio), "has_text": bool(user_text), "via":"flow-repeat", "prev_row_id": prev_id, "expect": prev_expect}
            return out

    best_text = None
    best_audioemb = None

    # 1) Texte → embeddings OpenAI → RPC match_audio_meta
    if user_text and openai_client:
        qvec = embed_text(user_text)
        if qvec:
            try:
                rpc = supabase.rpc("match_audio_meta", {
                    "p_lang": base_lang,
                    "p_query": qvec,
                    "p_limit": 5
                }).execute()
                cand = rpc.data or []
                if cand:
                    best = cand[0]
                    sim = 1.0 - float(best.get("distance", 1.0))
                    best_text = {"row": best, "via": "embed", "score": sim}
            except Exception as e:
                print("❌ rpc match (text) error:", e)

    # 2) Audio-only → embedding via worker → RPC match_audio_by_vector
    if inp.from_audio and not user_text and inp.audio:
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
                    b2 = cand2[0]
                    sim2 = 1.0 - float(b2.get("distance", 1.0))
                    best_audioemb = {"row": b2, "score": sim2, "via": "audio-embed"}
        except Exception as e:
            print("❌ audio-embed via worker error:", e)

    # 3) Appliquer les seuils
    if best_text and best_text["score"] < TEXT_SIM_THRESHOLD:
        best_text = None
    if best_audioemb and best_audioemb["score"] < AUDIO_SIM_THRESHOLD:
        best_audioemb = None

    # 4) Arbitrage — priorité texte > audio-embed
    final_hit, via = None, "fallback"
    cand_list = []
    if best_text: cand_list.append(best_text | {"prio": 2})
    if best_audioemb: cand_list.append(best_audioemb | {"prio": 1})
    if cand_list:
        cand_list.sort(key=lambda x: (x.get("prio",0), x.get("score",0)), reverse=True)
        top = cand_list[0]
        final_hit, via = {"row": top["row"], "score": top["score"]}, top.get("via","")

    # 5) Réponse défaut si rien
    if not final_hit:
        default_mina = "moudékoukou gnémousséwo"
        out = {"reply": default_mina, "row_id": None}
        if inp.debug:
            out["debug"] = {
                "from_audio": bool(inp.from_audio), "has_text": bool(user_text),
                "via": via, "baseLang": base_lang, "sourceLang": src, "targetLang": tgt,
                "best_text": None, "best_audio": None,
                "input": user_text, "chosen_row_id": None, "note": "no match → default"
            }
        try:
            supabase.table("server_logs").insert({
                "kind": "no_match",
                "input": user_text,
                "base_lang": base_lang,
                "debug": json.dumps(out.get("debug", {}))
            }).execute()
        except Exception:
            pass
        return out

    r = final_hit["row"]
    rs, fr, en, tx = (r.get("reply_same") or "").strip(), (r.get("fr") or "").strip(), (r.get("en") or "").strip(), (r.get("text") or "").strip()

    # Si ce nœud attend une réponse (flow)
    expect = (r.get("fsm_expect") or "none").lower()
    if expect in {"yn","choice"}:
        out = {
            "reply": pick_reply(src, tgt, tx, rs, fr, en, inp.mode),
            "row_id": r.get("id")
        }
        if inp.debug:
            out["debug"] = {
                "from_audio": bool(inp.from_audio), "has_text": bool(user_text),
                "via": via, "baseLang": base_lang, "sourceLang": src, "targetLang": tgt,
                "text_best": None if not best_text else {"row_id": best_text["row"].get("id"), "score": round(best_text["score"],4)},
                "audio_best": None if not best_audioemb else {"row_id": best_audioemb["row"].get("id"), "score": round(best_audioemb["score"],4)},
                "chosen_row_id": r.get("id"),
                "expect": expect
            }
        return out

    # Sinon : réponse simple “table only”
    reply = pick_reply(src, tgt, tx, rs, fr, en, inp.mode)
    out = {"reply": reply, "row_id": r.get("id")}
    if inp.debug:
        out["debug"] = {
            "from_audio": bool(inp.from_audio), "has_text": bool(user_text),
            "via": via, "baseLang": base_lang, "sourceLang": src, "targetLang": tgt,
            "text_best": None if not best_text else {
                "row_id": best_text["row"].get("id"), "via": best_text["via"], "score": round(best_text["score"],4)
            },
            "audio_best": None if not best_audioemb else {
                "row_id": best_audioemb["row"].get("id"), "score": round(best_audioemb["score"],4)
            },
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

# ------------------------- STT (Whisper non-streaming) -------------------------
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

# ------------------------- Nearby (OSM via Overpass, avec retry) -------------------------
# Miroirs Overpass
OVERPASS_URLS = [
    os.getenv("OVERPASS_URL", "").strip() or None,
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
OVERPASS_URLS = [u for u in OVERPASS_URLS if u]

OSM_TIMEOUT  = int(os.getenv("OSM_TIMEOUT", "25"))
NEARBY_LIMIT = int(os.getenv("NEARBY_LIMIT", "20"))

_KIND_QUERIES = {
    "pharmacy": [
        'node["amenity"="pharmacy"]',
        'way["amenity"="pharmacy"]',
        'relation["amenity"="pharmacy"]',
    ],
    "health": [
        'node["amenity"~"hospital|clinic|doctors"]',
        'way["amenity"~"hospital|clinic|doctors"]',
        'relation["amenity"~"hospital|clinic|doctors"]',
        'node["healthcare"]',
        'way["healthcare"]',
        'relation["healthcare"]',
    ],
    "food": [
        'node["amenity"~"restaurant|fast_food|cafe"]',
        'way["amenity"~"restaurant|fast_food|cafe"]',
        'relation["amenity"~"restaurant|fast_food|cafe"]',
    ],
}

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dl     = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _build_overpass_ql(kind: str, lat: float, lon: float, radius: int) -> str:
    parts = _KIND_QUERIES.get(kind, _KIND_QUERIES["pharmacy"])
    around = f"(around:{max(200, min(int(radius or 4000), 15000))},{lat},{lon})"
    body = "".join(f"{sel}{around};" for sel in parts)
    return f"""
[out:json][timeout:{OSM_TIMEOUT}];
(
  {body}
);
out center {NEARBY_LIMIT};
"""

def _extract_element_pos(el: dict):
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    c = el.get("center")
    if c and "lat" in c and "lon" in c:
        return float(c["lat"]), float(c["lon"])
    return None, None

def _best_name(tags: dict) -> str:
    if not tags: return "(sans nom)"
    for k in ("name:fr", "name:en", "name"):
        if tags.get(k): return tags[k]
    return tags.get("brand") or tags.get("healthcare") or "(sans nom)"

def _addr(tags: dict) -> str:
    if not tags: return ""
    bits = []
    for k in ("addr:street","addr:housenumber","addr:city","addr:district"):
        v = tags.get(k)
        if v: bits.append(v)
    return ", ".join(bits)

@app.post("/api/nearby")
def api_nearby(inp: NearbyIn):
    if not (-90 <= inp.lat <= 90 and -180 <= inp.lon <= 180):
        raise HTTPException(status_code=400, detail="Coordonnées invalides")

    kind = inp.kind if inp.kind in _KIND_QUERIES else "pharmacy"
    ql = _build_overpass_ql(kind, inp.lat, inp.lon, inp.radius)

    data = None
    last_err = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": ql}, timeout=OSM_TIMEOUT+5)
            if r.status_code in (429, 408) or r.status_code >= 500:
                last_err = f"{url} → HTTP {r.status_code}"
                continue
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = f"{url} → {repr(e)}"
            continue

    if not data:
        log_server("nearby_err", {"kind": kind, "err": last_err})
        raise HTTPException(status_code=502, detail="Overpass indisponible")

    items = []
    for el in data.get("elements", []):
        lat, lon = _extract_element_pos(el)
        if lat is None: 
            continue
        tags = el.get("tags", {}) or {}
        dist = int(round(_haversine_m(inp.lat, inp.lon, lat, lon)))
        name = _best_name(tags)
        addr = _addr(tags)
        cat  = tags.get("amenity") or tags.get("healthcare") or ""
        oh   = tags.get("opening_hours") or ""

        items.append({
            "name": name,
            "category": cat or None,
            "addr": addr or None,
            "lat": lat, "lon": lon,
            "distance_m": dist,
            "opening_hours": oh or None,
            "osm_id": f'{el.get("type","node")}/{el.get("id","")}',
        })

    items.sort(key=lambda x: x["distance_m"])
    items = items[:NEARBY_LIMIT]

    log_server("nearby_ok", {"kind": kind, "count": len(items), "lat": inp.lat, "lon": inp.lon, "radius": inp.radius})

    if not items:
        return {"items": [], "notice": "Aucun résultat trouvé à proximité."}

    return {"items": items}
