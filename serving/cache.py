# serving/cache.py
from __future__ import annotations
import os, time, json, hashlib
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:
    redis = None  # optional

def _sha1(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_key(namespace: str, payload: dict) -> str:
    return f"intelsent:{namespace}:{_sha1(payload)}"

class NullCache:
    def get(self, key: str) -> Optional[dict]: return None
    def set(self, key: str, value: dict) -> None: return None

class TTLCacheInProc:
    def __init__(self, ttl_s: int = 600):
        self.ttl = ttl_s
        self._store: dict[str, tuple[float, dict]] = {}

    def get(self, key: str) -> Optional[dict]:
        now = time.time()
        item = self._store.get(key)
        if not item: return None
        exp, val = item
        if exp < now:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: dict) -> None:
        self._store[key] = (time.time() + self.ttl, value)

class RedisCache:
    def __init__(self, url: str, ttl_s: int = 600):
        self.ttl = ttl_s
        self.r = redis.from_url(url, decode_responses=True)  # type: ignore

    def get(self, key: str) -> Optional[dict]:
        s = self.r.get(key)
        return json.loads(s) if s else None

    def set(self, key: str, value: dict) -> None:
        self.r.setex(key, self.ttl, json.dumps(value, separators=(",", ":")))

def get_cache(ttl_s: int = 600) -> object:
    if os.getenv("CACHE_ENABLED", "1").lower() not in {"1", "true", "yes"}:
        return NullCache()
    url = os.getenv("REDIS_URL", "").strip()
    if url and redis is not None:
        return RedisCache(url, ttl_s)
    return TTLCacheInProc(ttl_s)
