from fastapi import FastAPI

app = FastAPI(title="IntelSent API", version="0.0.1")

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"msg": "IntelSent skeleton ready. /docs for Swagger."}
