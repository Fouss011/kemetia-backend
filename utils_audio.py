import base64
from typing import Tuple

def dataurl_to_bytes(data_url: str) -> Tuple[bytes, str]:
    """
    Reçoit un dataURL "data:audio/webm;codecs=opus;base64,AAAA..."
    Retourne (bytes, mime)
    """
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")
    head, b64 = data_url.split(",", 1)
    mime = head.split(";")[0].split(":",1)[1] if ":" in head else "application/octet-stream"
    return base64.b64decode(b64), mime
