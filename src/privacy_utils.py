# src/privacy_utils.py

import hashlib

def sha256_hex(value: object) -> str:
    s = str(value).encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()
