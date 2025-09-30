import os
os.environ["SKIP_CHAIN_INIT"] = "1"  # prevent real DB init on import

from fastapi.testclient import TestClient
import importlib

api = importlib.import_module("serving.api")

class DummyChain:
    def __init__(self): self.use_openai = False
    def run(self, question, company=None, year=None, top_k=5):
        return {
            "answer": f"[stub] {question} ({company},{year})",
            "contexts": ["stub ctx"],
            "meta": {"company": company, "year": year},
        }

# inject stub
api.CHAIN = DummyChain()
client = TestClient(api.app)

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_openapi_root_exists():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def test_query_stubbed():
    payload = {"text":"hello","company":"MSFT","year":2022,"top_k":3,"no_openai":True}
    r = client.post("/query", json=payload)
    assert r.status_code == 200
    j = r.json()
    assert j["answer"].startswith("[stub] hello")
    assert j["contexts"] == ["stub ctx"]
