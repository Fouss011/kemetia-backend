# app.py
from __future__ import annotations
import os, re, json, base64, time, secrets, unicodedata
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

from supabase import create_client, Client
import requests

# ──────────────────────────────────────────────
# Init / Config
# ──────────────────────────────────────────────
load_dotenv()  # charge .env si présent

SUPABASE_URL = os.getenv("SUPABASE_URL") or ""
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or ""
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audio")  # ← crée un bucket public "audio"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""

# Seuils / pondérations
TEXT_SIM_THRESHOLD = float(os.getenv("TEXT_SIM_THRESHOLD", "0.68"))
MFCC_MIN_SIM       = float(os.getenv("MFCC_MIN_SCORE",  "0.94"))
MFCC_MIN_ROWS      = int(os.getenv("MFCC_MIN_ROWS",     "8"))
COMBO_TEXT_WEIGHT  = 0.60
COMBO_MFCC_WEIGHT  = 0.40
COMBO_BONUS_SAME   = 0.08

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("⚠️  SUPABASE_URL / SUPABASE_SERVICE_KEY manquants – /api/chat/collect/learn échoueront.")

supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    print("❌ Supabase init error:", e)
    supabase = None

app = FastAPI(title="Kemetia API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu peux restreindre à ton domaine Netlify si tu veux
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Utils texte
# ──────────────────────────────────────────────
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

def char_ngrams(s: str, n: int = 3) -> set:
    t = re.sub(r"[^a-z0-9]", "", nk(s))
    if len(t) <= n:
        return {t}
    return { t[i:i+n] for i in range(0, len(t)-n+1) }

def jaccard(a: str, b: str) -> float:
    A, B = char_ngrams(a), char_ngrams(b)
    inter = len(A & B)
    union = len(A | B)
    return (inter / union) if union else 0.0

def lev_ratio(a: str, b: str) -> float:
    s = re.sub(r"[^a-z0-9]", "", nk(a))
    t = re.sub(r"[^a-z0-9]", "", nk(b))
    m, n = len(s), len(t)
    if m == 0 and n == 0: return 1.0
    if m == 0 or n == 0:  return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s[i-1] == t[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    dist = dp[m][n]
    return 1.0 - dist / max(m, n)

def fuzzy(a: str, b: str) -> float:
    return max(jaccard(a,b), lev_ratio(a,b))

# ──────────────────────────────────────────────
# Utils MFCC
# ──────────────────────────────────────────────
def as_vec13(v: Any) -> Optional[np.ndarray]:
    try:
        arr = list(v) if isinstance(v, (list, tuple)) else []
        if len(arr) < 10: 
            return None
        x = np.array(arr[:13], dtype=float)
        return x
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
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
    d = cosine_dist(a, b)
    return 1.0 - float(d)

# ──────────────────────────────────────────────
# Utils Storage (upload audio → Supabase Storage)
# ──────────────────────────────────────────────
def _storage_public_url(path: str) -> str:
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/storage/v1/object/public/{SUPABASE_BUCKET}/{path.lstrip('/')}"

def _random_name(prefix: str = "rec", ext: str = "webm") -> str:
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(6)
    return f"{prefix}-{ts}-{rand}.{ext}"

def _guess_ext_from_mime(mime: str) -> str:
    if not mime: return "webm"
    m = mime.lower()
    if "ogg" in m: return "ogg"
    if "mp4" in m: return "m4a"
    if "mpeg" in m or "mp3" in m: return "mp3"
    if "wav" in m: return "wav"
    return "webm"

def upload_bytes_to_storage(data: bytes, mime: str, *, prefix: str="recs") -> tuple[str,str]:
    """
    Upload binaire vers Supabase Storage (bucket public).
    Retourne (public_url, storage_path).
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase non configuré")

    ext = _guess_ext_from_mime(mime)
    fname = _random_name(prefix=prefix, ext=ext)
    path = f"{prefix}/{fname}"

    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": mime or "application/octet-stream",
            "x-upsert": "true",
            "Cache-Control": "public, max-age=31536000",
        },
        data=data,
        timeout=30,
    )
    if r.status_code >= 300:
        raise HTTPException(status_code=r.status_code, detail=f"upload storage failed: {r.text}")
    return _storage_public_url(path), path

# ──────────────────────────────────────────────
# Schémas requêtes
# ──────────────────────────────────────────────
class ChatIn(BaseModel):
    text: str = ""
    mode: str = "exchange"          # "exchange" | "translate"
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
    audio: Optional[Dict[str, Any]] = None  # { "dataURL": "...", "mime": "...", "filename": "opt" }
    duration_ms: Optional[int] = None
    mfcc: Optional[Dict[str, Any]] = None

class LearnIn(BaseModel):
    row_id: int
    accepted: bool
    input_type: str = "audio"            # "audio" | "text"
    user_mfcc: Optional[Dict[str, Any]] = None
    correction_text: Optional[str] = None
    correction_row_id: Optional[int] = None

class NearbyIn(BaseModel):
    kind: str = "pharmacy"               # "pharmacy" | "health" | "food"
    lat: float
    lon: float
    radius: int = 4000

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.post("/api/chat")
def api_chat(inp: ChatIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    user_text = (inp.text or "").strip()

    # baseLang (si mode translate depuis FR → cible locale)
    base_lang = inp.sourceLang.lower() if inp.sourceLang.lower() != "fr" else inp.targetLang.lower()
    if base_lang not in {"mina","bm","ee","ha","sw","fr","en"}:
        base_lang = "mina"

    # fetch dataset
    q = (supabase.table("audio_meta")
         .select("id,lang,category,text,reply_same,fr,en,variants_text,variants_sig,filename,url,created_at,mfcc")
         .eq("lang", base_lang)
         .order("created_at", desc=True)
         .limit(5000))
    rows = q.execute().data or []

    # index texte
    def split_variants(v: Optional[str]) -> List[str]:
        if not v: return []
        return [s.strip() for s in re.split(r"[\n,]+", v) if s.strip()]

    user_is_french = inp.sourceLang.lower() == "fr"
    def forms_for_row(r: Dict[str, Any]) -> List[Dict[str, str]]:
        forms = []
        if not user_is_french:
            for v in [r.get("text","")] + split_variants(r.get("variants_text")) + split_variants(r.get("variants_sig")):
                s = soft_canon(v)
                if s: forms.append({"raw": v, "sig": s, "sigNS": s.replace(" ","")})
        else:
            for v in split_variants(r.get("fr")):
                s = soft_canon(v)
                if s: forms.append({"raw": v, "sig": s, "sigNS": s.replace(" ","")})
        # dedup
        seen, uniq = set(), []
        for f in forms:
            key = f["sig"] + "|" + f["sigNS"]
            if key in seen: continue
            seen.add(key); uniq.append(f)
        return uniq

    sig_user   = soft_canon(user_text)
    sig_userNS = sig_user.replace(" ","")
    best_text = None

    if user_text:
        # exact
        for r in rows:
            for f in forms_for_row(r):
                if f["sig"] == sig_user or f["sigNS"] == sig_userNS:
                    best_text = {"row": r, "via": "exact", "score": 1.0, "matched": user_text}
                    break
            if best_text: break

        # fuzzy
        if not best_text:
            for r in rows:
                for f in forms_for_row(r):
                    sc = fuzzy(user_text, f["raw"])
                    if not best_text or sc > best_text["score"]:
                        best_text = {"row": r, "via": "fuzzy", "score": sc, "matched": f["raw"]}
            if best_text and best_text["score"] < TEXT_SIM_THRESHOLD:
                best_text = None

    # MFCC
    mfcc_rows = sum(1 for r in rows if row_mfcc_vec(r) is not None)
    best_mfcc = None
    uvec = user_mfcc_vec(inp.mfcc) if (inp.from_audio and inp.mfcc) else None
    if inp.from_audio and uvec is not None and mfcc_rows >= MFCC_MIN_ROWS:
        for r in rows:
            v = row_mfcc_vec(r)
            if v is None: continue
            sc = cosine_sim(uvec, v)
            if (best_mfcc is None) or (sc > best_mfcc["score"]):
                best_mfcc = {"row": r, "score": sc, "via": "mfcc-only"}
        if best_mfcc and best_mfcc["score"] < MFCC_MIN_SIM:
            best_mfcc = None
        # garde-fou "Salutations"
        if best_mfcc and (str(best_mfcc["row"].get("category","")).lower() == "salutations"):
            if best_mfcc["score"] < (MFCC_MIN_SIM + 0.0015):
                best_mfcc = None

    # Choix final
    final_hit, via = None, "fallback"
    if best_text and best_mfcc:
        if best_text["row"]["id"] == best_mfcc["row"]["id"]:
            final_hit = {
                "row": best_text["row"],
                "score": COMBO_TEXT_WEIGHT*best_text["score"] + COMBO_MFCC_WEIGHT*best_mfcc["score"] + COMBO_BONUS_SAME
            }
            via = "combo-mfcc-same"
        else:
            tZ = best_text["score"] - TEXT_SIM_THRESHOLD
            mZ = best_mfcc["score"] - MFCC_MIN_SIM
            if mZ > tZ:
                final_hit, via = {"row": best_mfcc["row"], "score": best_mfcc["score"]}, "mfcc-only-override"
            else:
                final_hit, via = {"row": best_text["row"],  "score": best_text["score"]},  "text-priority"
    elif best_text:
        final_hit, via = best_text, best_text["via"]
    elif best_mfcc:
        final_hit, via = best_mfcc, "mfcc-only"

    if not final_hit:
        out = {"reply": "", "row_id": None}
        if inp.debug:
            out["debug"] = {
                "input": user_text, "sigUser": sig_user, "baseLang": base_lang, "via": via,
                "text_best": None if not best_text else {
                    "row_id": best_text["row"]["id"], "via": best_text["via"],
                    "score": round(best_text["score"],4), "matched": best_text.get("matched")
                },
                "mfcc_best": None if not best_mfcc else {
                    "row_id": best_mfcc["row"]["id"], "score": round(best_mfcc["score"],4)
                },
                "mfcc_rows": mfcc_rows, "chosen_row_id": None
            }
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
            "input": user_text, "sigUser": sig_user, "baseLang": base_lang, "via": via,
            "text_best": None if not best_text else {
                "row_id": best_text["row"]["id"], "via": best_text["via"],
                "score": round(best_text["score"],4), "matched": best_text.get("matched")
            },
            "mfcc_best": None if not best_mfcc else {
                "row_id": best_mfcc["row"]["id"], "score": round(best_mfcc["score"],4)
            },
            "mfcc_rows": mfcc_rows, "chosen_row_id": r.get("id")
        }
    return out


@app.post("/api/collect")
def api_collect(inp: CollectIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    file_url = None
    filename = (inp.filename or "").strip() or None
    mime = (inp.mime or "").strip() or "audio/webm"

    # Upload audio si présent en dataURL
    if inp.audio and isinstance(inp.audio, dict) and inp.audio.get("dataURL"):
        try:
            dataURL = inp.audio["dataURL"]
            head, b64 = dataURL.split(",", 1)
            # mime prioritaire: champ fourni > dataURL
            if not inp.mime:
                try:
                    mime = head.split(";")[0].split(":",1)[1]
                except:
                    mime = "application/octet-stream"
            raw = base64.b64decode(b64)
            file_url, storage_path = upload_bytes_to_storage(raw, mime, prefix="recs")
            filename = storage_path.split("/")[-1]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"upload audio failed: {e}")

    row = {
        "lang": inp.lang.lower(),
        "category": (inp.category or "").strip() or None,
        "text": (inp.text or "").strip(),
        "variants_text": (inp.variants or "").strip() or None,
        "reply_same": (inp.reply_same or "").strip() or None,
        "fr": (inp.fr or "").strip() or None,
        "en": (inp.en or "").strip() or None,
        "filename": filename,
        "mime": mime,
        "duration_ms": inp.duration_ms,
        "url": file_url,
        "mfcc": json.loads(json.dumps(inp.mfcc)) if inp.mfcc else None,
    }
    if not row["text"]:
        raise HTTPException(status_code=400, detail="text obligatoire")

    res = supabase.table("audio_meta").insert(row).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="Insert échoué")
    return {"ok": True, "id": res.data[0]["id"], "url": file_url, "filename": filename}


@app.post("/api/learn")
def api_learn(inp: LearnIn):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase non configuré")

    meta = { "ip": None, "ua": "fastapi" }
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


@app.post("/api/stt")
def api_stt(payload: Dict[str, Any]):
    """
    Attendu: { audio: dataURL (base64), mime: str, lang: "fr"|"en"|... }
    Si OPENAI_API_KEY absent → 501 (non configuré).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=501, detail="STT non configuré (OPENAI_API_KEY manquant)")
    # TODO: branche Whisper/OpenAI ici si tu veux une vraie transcription
    return {"text": ""}


@app.post("/api/nearby")
def api_nearby(inp: NearbyIn):
    # Stub : renvoie vide (branche ton fournisseur plus tard)
    return {"items": [], "notice": "Aucun résultat dans ce rayon."}
