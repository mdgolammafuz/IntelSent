# serving/api.py
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="IntelSent API", version="0.0.1")

# expose /metrics automatically
Instrumentator().instrument(app).expose(app)

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"msg": "IntelSent skeleton ready. /docs for Swagger."}
