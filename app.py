

### backend/app.py
# app.py  (FastAPI complet, embeddings OpenAI + STT Whisper + MFCC fallback)
from __future__ import annotations
import os, re, json, base64, tempfile, unicodedata
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from supabase import create_client, Client

# OpenAI (>=1.x)
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TEXT_SIM_THRESHOLD = float(os.getenv("TEXT_SIM_THRESHOLD", "0.68"))
MFCC_MIN_SIM       = float(os.getenv("MFCC_MIN_SCORE",  "0.94"))
MFCC_MIN_ROWS      = int(os.getenv("MFCC_MIN_ROWS",     "8"))
COMBO_TEXT_WEIGHT  = 0.60
COMBO_MFCC_WEIGHT  = 0.40
COMBO_BONUS_SAME   = 0.08

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

app = FastAPI(title="Kemetia API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utils texte ----------
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

# ---------- Utils MFCC (fallback audio-only) ----------
def as_vec13(v: Any) -> Optional[np.ndarray]:
    try:
        arr = list(v) if isinstance(v, (list, tuple)) else []
        if len(arr) < 10:
            return None
        return np.array(arr[:13], dtype=float)
    except Exception:
        return None

def row_mfcc_vec(row: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        m = row.get("mfcc")
        if isinstance(m, str):
            m = json.loads(m)
        mean = (((m or {}).get("centroid") or [{}])[0] or {}).get("mean")
        return as_vec13(mean)
    except Exception:
        return None

def user_mfcc_vec(mfcc: Any) -> Optional[np.ndarray]:
    try:
        m = mfcc
        if isinstance(m, str):
            m = json.loads(m)
        mean = (((m or {}).get("centroid") or [{}])[0] or {}).get("mean")
        return as_vec13(mean)
    except Exception:
        return None

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return 0.0
    n = min(a.shape[0], b.shape[0])
    d = cosine_dist(a[:n], b[:n])
    return 1.0 - float(d)

# ---------- OpenAI Embeddings ----------
import time
from functools import lru_cache

# Embedding model
EMB_MODEL = "text-embedding-3-small"

# Backoff state
EMBED_FAIL_UNTIL = 0.0

# Petit cache en mémoire pour éviter appels répétés sur le même texte
@lru_cache(maxsize=2048)
def _cached_embedding(q: str):
    # note: retourne un tuple (hashable pour lru_cache)
    resp = openai_client.embeddings.create(model=EMB_MODEL, input=q)
    vec = resp.data[0].embedding
    return tuple(vec)

def embed_text(text: str) -> Optional[List[float]]:
    """
    Retourne une liste de floats (ou None).
    Utilise un cache lru et un backoff simple si l'API renvoie une erreur (quota / rate limit).
    """
    global EMBED_FAIL_UNTIL
    if not openai_client:
        return None
    now = time.time()
    if now < EMBED_FAIL_UNTIL:
        # On est en backoff
        return None

    q = soft_canon(text)
    if not q:
        return None

    try:
        vec_tuple = _cached_embedding(q)
        return list(vec_tuple)
    except Exception as e:
        s = str(e) or ""
        # si quota / rate limit → backoff plus long
        if "RateLimit" in s or "insufficient_quota" in s or "quota" in s.lower() or "429" in s:
            EMBED_FAIL_UNTIL = time.time() + 120  # 2 min
        else:
            EMBED_FAIL_UNTIL = time.time() + 30   # 30s pour autres erreurs temporaires
        print("❌ embed error (backoff set):", repr(e))
        return None


# ---------- Schemas ----------
class ChatIn(BaseModel):
    text: str = ""
    mode: str = "exchange"
    sourceLang: str = "mina"
    targetLang: str = "fr"
    debug: bool = True
    bridge: bool = False
    from_audio: bool = False
    mfcc: Optional[Dict[str, Any]] = None
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
    mfcc: Optional[Dict[str, Any]] = None

class LearnIn(BaseModel):
    row_id: int
    accepted: bool
    input_type: str = "audio"
    user_mfcc: Optional[Dict[str, Any]] = None
    correction_text: Optional[str] = None
    correction_row_id: Optional[int] = None

class NearbyIn(BaseModel):
    kind: str = "pharmacy"
    lat: float
    lon: float
    radius: int = 4000

# ---------- Health ----------
@app.get("/health")
def health(): return {"ok": True}

# ---------- Chat ----------
@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    # 0) STT-first : si audio fourni et pas de texte, on transcrit (sync non-stream)
    user_text = (inp.text or "").strip()
    if not user_text and inp.from_audio and (inp.audio or inp.mfcc):
        if inp.audio and openai_client:
            try:
                header, b64 = inp.audio.split(",", 1)
                raw = base64.b64decode(b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                    f.write(raw)
                    tmp_path = f.name
                try:
                    with open(tmp_path, "rb") as fh:
                        tr = openai_client.audio.transcriptions.create(model="whisper-1", file=fh)
                    user_text = (tr.text or "").strip()
                finally:
                    try: os.remove(tmp_path)
                    except: pass
            except Exception as e:
                # On log l'erreur mais on continue : on fera fallback MFCC si STT échoue
                print("❌ STT in api_chat failed:", repr(e))

    # base language
    base_lang = inp.sourceLang.lower() if inp.sourceLang.lower() != "fr" else inp.targetLang.lower()
    if base_lang not in {"mina","bm","ee","ha","sw","fr","en"}:
        base_lang = "mina"

    # 1) Si texte -> Embeddings OpenAI + pgvector (RPC match_audio_meta)
    best_text = None
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
                    best_text = {
                        "row": best,
                        "via": "embed",
                        "score": sim,
                        "matched": None
                    }
            except Exception as e:
                print("❌ rpc match error:", e)

    # 2) Fallback MFCC-only (si from_audio + pas de texte fiable)
    best_mfcc = None
    rows = []
    if inp.from_audio and (inp.mfcc is not None):
        q = (supabase.table("audio_meta")
             .select("id,lang,category,text,reply_same,fr,en,variants_text,variants_sig,filename,url,created_at,mfcc")
             .eq("lang", base_lang)
             .order("created_at", desc=True)
             .limit(5000))
        rows = q.execute().data or []
        mfcc_rows = sum(1 for r in rows if row_mfcc_vec(r) is not None)
        uvec = user_mfcc_vec(inp.mfcc)
        if uvec is not None and mfcc_rows >= MFCC_MIN_ROWS:
            for r in rows:
                v = row_mfcc_vec(r)
                if v is None: continue
                sc = cosine_sim(uvec, v)
                if (best_mfcc is None) or (sc > best_mfcc["score"]):
                    best_mfcc = {"row": r, "score": sc, "via": "mfcc-only"}
            if best_mfcc and best_mfcc["score"] < MFCC_MIN_SIM:
                best_mfcc = None
            if best_mfcc and (str(best_mfcc["row"].get("category","")).lower() == "salutations"):
                if best_mfcc["score"] < (MFCC_MIN_SIM + 0.0015):
                    best_mfcc = None
    else:
        mfcc_rows = 0

    # 3) Arbitrage final
    final_hit, via = None, "fallback"
    if best_text and best_mfcc:
        if best_text["row"].get("id") == best_mfcc["row"].get("id"):
            final_hit = {
                "row": best_text["row"],
                "score": COMBO_TEXT_WEIGHT*best_text["score"] + COMBO_MFCC_WEIGHT*best_mfcc["score"] + COMBO_BONUS_SAME
            }
            via = "combo-mfcc-same"
        else:
            tZ = (best_text["score"] - 0.40)
            mZ = (best_mfcc["score"] - MFCC_MIN_SIM)
            if mZ > tZ:
                final_hit, via = {"row": best_mfcc["row"], "score": best_mfcc["score"]}, "mfcc-only-override"
            else:
                final_hit, via = {"row": best_text["row"],  "score": best_text["score"]},  "embed-priority"
    elif best_text:
        final_hit, via = best_text, best_text["via"]
    elif best_mfcc:
        final_hit, via = best_mfcc, "mfcc-only"

    # 4) Réponse (fallback explicit)
    if not final_hit:
        default_mina = "moudékoukou gnémousséwo"
        out = {"reply": default_mina, "row_id": None}
        if inp.debug:
            out["debug"] = {
                "input": user_text,
                "baseLang": base_lang,
                "via": via,
                "text_best": None if not best_text else {
                    "row_id": best_text["row"].get("id"), "via": best_text["via"], "score": round(best_text["score"],4)
                },
                "mfcc_best": None if not best_mfcc else {
                    "row_id": best_mfcc["row"].get("id"), "score": round(best_mfcc["score"],4)
                },
                "mfcc_rows": mfcc_rows, "chosen_row_id": None,
                "note": "no match → returning default mina phrase"
            }
        # log léger en base (échec de matching)
        try:
            if supabase is not None:
                supabase.table("server_logs").insert({
                    "kind":"no_match",
                    "input": user_text,
                    "base_lang": base_lang,
                    "debug": json.dumps(out.get("debug", {}))
                }).execute()
        except Exception:
            pass
        return out

    r = final_hit["row"]
    rs, fr, en, tx = (r.get("reply_same") or "").strip(), (r.get("fr") or "").strip(), (r.get("en") or "").strip(), (r.get("text") or "").strip()
    src = (inp.sourceLang or "mina").lower()
    tgt = (inp.targetLang or "fr").lower()
    reply = ""
    if inp.mode == "exchange" or tgt == "same":
        reply = rs or tx or fr or en or ""
    else:
        if src == "fr":
            if tgt in {"mina","bm","ee","ha","sw"}:
                reply = tx or rs or en or fr or ""
            elif tgt == "en":
                reply = en or fr or tx or rs or ""
            else:
                reply = fr or ""
        else:
            if   tgt == "fr": reply = fr or tx or rs or en or ""
            elif tgt == "en": reply = en or fr or tx or rs or ""
            elif tgt == src:  reply = tx or rs or fr or en or ""
            else:             reply = tx or rs or fr or en or ""

    out = {"reply": reply, "row_id": r.get("id")}
    if inp.debug:
        out["debug"] = {
            "input": user_text, "baseLang": base_lang, "via": via,
            "text_best": None if not best_text else {"row_id": best_text["row"].get("id"), "via": best_text["via"], "score": round(best_text["score"],4)},
            "mfcc_best": None if not best_mfcc else {"row_id": best_mfcc["row"].get("id"), "score": round(best_mfcc["score"],4)},
            "mfcc_rows": mfcc_rows, "chosen_row_id": r.get("id")
        }
    return out


# ---------- Collect ----------
@app.post("/api/collect")
def api_collect(inp: CollectIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    row = {
        "lang": inp.lang.lower(),
        "category": (inp.category or "").strip() or None,
        "text": (inp.text or "").strip(),
        "variants_text": (inp.variants or "").strip() or None,
        "reply_same": (inp.reply_same or "").strip() or None,
        "fr": (inp.fr or "").strip() or None,
        "en": (inp.en or "").strip() or None,
        "filename": (inp.filename or "").strip() or None,
        "mime": (inp.mime or "").strip() or None,
        "duration_ms": inp.duration_ms,
        "mfcc": json.loads(json.dumps(inp.mfcc)) if inp.mfcc else None,
    }
    if not row["text"]:
        raise HTTPException(status_code=400, detail="text obligatoire")

    vec = embed_text(row["text"])
    if vec:
        row["embedding"] = vec
        row["embedding_generated"] = True


    res = supabase.table("audio_meta").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert échoué")
    return {"ok": True, "id": res.data[0]["id"]}

# ---------- Learn ----------
@app.post("/api/learn")
def api_learn(inp: LearnIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")
    meta = {"ip": None, "ua": "fastapi"}
    row = {
        "row_id": inp.row_id,
        "accepted": bool(inp.accepted),
        "input_type": inp.input_type,
        "user_mfcc": json.loads(json.dumps(inp.user_mfcc)) if inp.user_mfcc else None,
        "correction_text": (inp.correction_text or None),
        "correction_row_id": inp.correction_row_id,
        "meta": meta
    }
    res = supabase.table("events").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert events échoué")
    return {"ok": True, "inserted": res.data[0]}

# ---------- STT (Whisper non-streaming) ----------
@app.post("/api/stt")
def api_stt(payload: Dict[str, Any]):
    if not openai_client:
        raise HTTPException(status_code=501, detail="STT non configuré (OPENAI_API_KEY manquant)")
    data_url = payload.get("audio") or ""
    if not data_url.startswith("data:"):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    try:
        header, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio invalide")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
        f.write(raw)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as fh:
            tr = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=fh
            )
        text = (tr.text or "").strip()
        return {"text": text}
    except Exception as e:
        print("❌ STT error:", e)
        raise HTTPException(status_code=500, detail="STT error")
    finally:
        try: os.remove(tmp_path)
        except: pass

# ---------- Nearby (placeholder)
@app.post("/api/nearby")
def api_nearby(inp: NearbyIn):
    return {"items": [], "notice": "Aucun résultat dans ce rayon."}

