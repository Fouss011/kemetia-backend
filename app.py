# app.py — FastAPI (texte embeddings OpenAI + STT Whisper + fallback audio-embeddings OpenL3)
from __future__ import annotations

import os, re, json, base64, tempfile, unicodedata, subprocess, time
from functools import lru_cache
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import subprocess, tempfile
import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from supabase import create_client, Client

# OpenAI (>=1.x)
from openai import OpenAI

# Audio embedding deps
import soundfile as sf
import openl3
# === Audio embeddings (OpenL3) – imports ===




# ------------------------- Config & clients -------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Toggle pour activer l'endpoint audio-embedding
USE_AUDIO_EMB = os.getenv("USE_AUDIO_EMB", "0") == "1"


TEXT_SIM_THRESHOLD = float(os.getenv("TEXT_SIM_THRESHOLD", "0.68"))
COMBO_TEXT_WEIGHT  = 0.60  # (gardé si besoin d'évoluer plus tard)
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

# Toujours mettre des headers CORS même si exception
@app.middleware("http")
async def ensure_cors_headers(request, call_next):
    try:
        resp = await call_next(request)
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            {"detail": "server error", "error": str(e)},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type,Authorization",
            },
        )
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
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

# ---------- Audio-embedding helpers (OpenL3) ----------
def _decode_dataurl_to_wav_path(data_url: str) -> str:
    """
    Convertit une dataURL (webm/ogg/wav) en WAV mono 16k, retourne le chemin.
    Nécessite ffmpeg présent sur la machine.
    """
    if not data_url.startswith("data:"):
        raise ValueError("audio dataURL requis")
    header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(raw)
        src_path = f.name
    wav_path = src_path + ".wav"
    # transcodage → wav mono 16k
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try: os.remove(src_path)
    except: pass
    return wav_path

def _compute_openl3_embedding(wav_path: str) -> list[float]:
    """
    Calcule un embedding audio OpenL3 (mean-pooling) et renvoie une liste de floats.

    Env optionnelles :
      AUDIO_EMB_CONTENT_TYPE ∈ {"music","env"}  (defaut: "music")
      AUDIO_EMB_INPUT_REPR   ∈ {"mel256","mel128"} (defaut: "mel256")
      AUDIO_EMB_EMBED_SIZE   ∈ {"512","6144"}   (defaut: "512")
    """
    import os
    import numpy as np
    import soundfile as sf
    import openl3

    # Lire le WAV (déjà transcodé en mono/16k en amont)
    try:
        y, sr = sf.read(wav_path, always_2d=False)
    except Exception as e:
        raise RuntimeError(f"audio read error: {e}")

    if y is None:
        try: os.remove(wav_path)
        except: pass
        return []
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:  # stéréo → mono
        y = np.mean(y, axis=1)
    if not y.size or not np.isfinite(y).any():
        try: os.remove(wav_path)
        except: pass
        return []

    # Params via env (avec valeurs sûres)
    ct = os.getenv("AUDIO_EMB_CONTENT_TYPE", "music").lower()
    if ct not in ("music", "env"):
        ct = "music"

    inp = os.getenv("AUDIO_EMB_INPUT_REPR", "mel256").lower()
    if inp not in ("mel256", "mel128"):
        inp = "mel256"

    es = os.getenv("AUDIO_EMB_EMBED_SIZE", "512")
    try:
        es_int = int(es)
    except Exception:
        es_int = 512
    if es_int not in (512, 6144):
        es_int = 512

    # Calcul OpenL3
    try:
        emb, _ = openl3.get_audio_embedding(
            y, sr,
            input_repr=inp,
            content_type=ct,      # "music" ou "env" (pas "speech")
            embedding_size=es_int
        )
    except Exception as e:
        try: os.remove(wav_path)
        except: pass
        raise RuntimeError(f"openl3 error: {e}")

    # Mean-pooling temporel
    if emb is None or not hasattr(emb, "shape") or emb.shape[0] == 0:
        try: os.remove(wav_path)
        except: pass
        return []
    vec = np.mean(emb, axis=0)
    vec = np.where(np.isfinite(vec), vec, 0.0).astype(float)

    try: os.remove(wav_path)
    except: pass

    return vec.tolist()


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
        if "RateLimit" in s or "insufficient_quota" in s or "quota" in s.lower() or "429" in s:
            EMBED_FAIL_UNTIL = time.time() + 120
        else:
            EMBED_FAIL_UNTIL = time.time() + 30
        print("❌ embed error (backoff set):", repr(e))
        return None


# ------------------------- Audio embeddings (OpenL3) -------------------------
def _decode_dataurl_to_wav_path(data_url: str) -> str:
    """Transcode dataURL (webm/ogg/wav) -> WAV mono 16k, return path."""
    if not data_url or not data_url.startswith("data:"):
        raise ValueError("audio dataURL requis")
    header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".in") as f:
        f.write(raw)
        src_path = f.name
    wav_path = src_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", wav_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        try: os.remove(src_path)
        except: pass
        raise RuntimeError(f"ffmpeg transcode échec: {e}")
    try: os.remove(src_path)
    except: pass
    return wav_path

def _compute_openl3_embedding(wav_path: str) -> List[float]:
    """Compute OpenL3 512-dim embedding with mean pooling."""
    try:
        y, sr = sf.read(wav_path, always_2d=False)
        emb, _ = openl3.get_audio_embedding(
            y, sr, embedding_size=512, input_repr="mel256", content_type="speech"
        )
        if emb is None or getattr(emb, "shape", [0])[0] == 0:
            return []
        vec = np.mean(emb, axis=0).astype(float).tolist()
        return vec
    except Exception as e:
        raise RuntimeError(f"openl3 error: {e}")
    finally:
        try: os.remove(wav_path)
        except: pass


# ------------------------- Schemas -------------------------
class ChatIn(BaseModel):
    text: str = ""
    mode: str = "exchange"     # "exchange" | "translate"
    sourceLang: str = "mina"
    targetLang: str = "fr"
    debug: bool = True
    bridge: bool = False
    from_audio: bool = False
    audio: Optional[str] = None   # dataURL audio (pour fallback audio-embed serveur)
    mfcc: Optional[Dict[str, Any]] = None  # conservé pour compat mais ignoré
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


# ------------------------- Health -------------------------
@app.get("/health")
def health():
    return {"ok": True}


# ------------------------- Chat -------------------------
@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    user_text = (inp.text or "").strip()

    base_lang = inp.sourceLang.lower() if inp.sourceLang.lower() != "fr" else inp.targetLang.lower()
    if base_lang not in {"mina","bm","ee","ha","sw","fr","en"}:
        base_lang = "mina"

    best_text = None
    best_audioemb = None
    mfcc_rows = 0  # gardé pour debug

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
                print("❌ rpc match error:", e)

    # 2) Fallback audio-embedding si pas de texte mais audio présent
    if inp.from_audio and not user_text and inp.audio:
        try:
            wav_path = _decode_dataurl_to_wav_path(inp.audio)
            avec = _compute_openl3_embedding(wav_path)
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
            print("❌ audio-embed fallback error:", e)

    # 3) Arbitrage final — priorité texte > audio-embed
    final_hit, via = None, "fallback"
    cand_list = []
    if best_text: cand_list.append(best_text | {"prio": 2})
    if best_audioemb: cand_list.append(best_audioemb | {"prio": 1})
    if cand_list:
        cand_list.sort(key=lambda x: (x.get("prio",0), x.get("score",0)), reverse=True)
        top = cand_list[0]
        final_hit, via = {"row": top["row"], "score": top["score"]}, top.get("via","")

    # 4) Réponse
    if not final_hit:
        default_mina = "moudékoukou gnémousséwo"
        out = {"reply": default_mina, "row_id": None}
        if inp.debug:
            out["debug"] = {
                "input": user_text, "baseLang": base_lang, "via": via,
                "text_best": None if not best_text else {
                    "row_id": best_text["row"].get("id"), "via": best_text["via"], "score": round(best_text["score"],4)
                },
                "audio_best": None if not best_audioemb else {
                    "row_id": best_audioemb["row"].get("id"), "score": round(best_audioemb["score"],4)
                },
                "mfcc_rows": mfcc_rows, "chosen_row_id": None,
                "note": "no match → returning default mina phrase"
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
    src = (inp.sourceLang or "mina").lower()
    tgt = (inp.targetLang or "fr").lower()

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
            "text_best": None if not best_text else {
                "row_id": best_text["row"].get("id"), "via": best_text["via"], "score": round(best_text["score"],4)
            },
            "audio_best": None if not best_audioemb else {
                "row_id": best_audioemb["row"].get("id"), "score": round(best_audioemb["score"],4)
            },
            "mfcc_rows": mfcc_rows, "chosen_row_id": r.get("id")
        }
    return out


# ------------------------- Collect -------------------------
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

    # Embedding texte (OpenAI)
    vec = embed_text(row["text"])
    if vec:
        row["embedding"] = vec

    # audio_embedding si audio fourni
    audio_embedding = None
    if inp.audio:
        try:
            wav_path = _decode_dataurl_to_wav_path(inp.audio)
            avec = _compute_openl3_embedding(wav_path)
            if avec:
                audio_embedding = avec
        except Exception as e:
            print("⚠️ audio_embedding compute error:", e)

    if audio_embedding:
        row["audio_embedding"] = audio_embedding

    res = supabase.table("audio_meta").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert échoué")
    return {"ok": True, "id": res.data[0]["id"]}


# ------------------------- Learn -------------------------
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


# ------------------------- STT (Whisper non-streaming) -------------------------
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


# ------------------------- Audio-embedding direct (outil) -------------------------
@app.post("/api/compute_audio_embedding")
def api_compute_audio_embedding(payload: Dict[str, Any]):
    data_url = payload.get("audio") or ""
    if not data_url.startswith("data:"):
        raise HTTPException(status_code=400, detail="audio dataURL requis")
    try:
        wav_path = _decode_dataurl_to_wav_path(data_url)
        vec = _compute_openl3_embedding(wav_path)
        if not vec:
            raise HTTPException(status_code=500, detail="embedding vide")
        return {"embedding": vec, "dim": len(vec)}
    except HTTPException:
        raise
    except Exception as e:
        print("❌ audio-embed error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Audio-embedding endpoint ----------
@app.post("/api/compute_audio_embedding")
def api_compute_audio_embedding(payload: dict):
    # sécurité: on peut désactiver par env si besoin
    if not USE_AUDIO_EMB:
        raise HTTPException(status_code=501, detail="audio-embedding désactivé (set USE_AUDIO_EMB=1)")

    data_url = (payload or {}).get("audio") or ""
    if not data_url.startswith("data:"):
        raise HTTPException(status_code=400, detail="audio dataURL requis")

    try:
        wav_path = _decode_dataurl_to_wav_path(data_url)
        vec = _compute_openl3_embedding(wav_path)
        if not vec:
            raise HTTPException(status_code=500, detail="embedding vide")
        return {"embedding": vec, "dim": len(vec)}
    except HTTPException:
        raise
    except Exception as e:
        print("❌ audio-embed error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------- Nearby (stub) -------------------------
@app.post("/api/nearby")
def api_nearby(inp: NearbyIn):
    return {"items": [], "notice": "Aucun résultat dans ce rayon."}
