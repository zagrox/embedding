import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware
import threading  # <-- NEW

CACHE_DIR = "/app/fastembed_cache"
os.environ["FASTEMBED_CACHE_PATH"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_loaded = False

class EmbedRequest(BaseModel):
    text: str

# Lock to serialize embedding calls
embed_lock = threading.Lock()  # <-- NEW

@app.on_event("startup")
def startup_event():
    global model, model_loaded
    try:
        print("🔄 Loading embedding model on startup...")
        model = TextEmbedding(MODEL_NAME)
        # warm up to force full load
        _ = list(model.embed(["warmup"]))
        model_loaded = True
        print("✅ Model loaded and warmed up.")
    except Exception as e:
        print("❌ CRITICAL ERROR LOADING MODEL ON STARTUP")
        traceback.print_exc()
        model_loaded = False

@app.get("/")
def root():
    return {
        "status": "active",
        "model": MODEL_NAME,
        "cache_dir": CACHE_DIR,
        "model_loaded": model_loaded,
    }

# ===== This is the ONLY endpoint you need to change =====
@app.post("/embed")
def embed_text(req: EmbedRequest):
    global model, model_loaded
    with embed_lock:  # <-- NEW: only one embedding at a time
        try:
            if not model_loaded or model is None:
                startup_event()
            vectors = list(model.embed([req.text]))
            return {"embedding": vectors[0]}
        except Exception as e:
            print("❌ CRITICAL ERROR DURING EMBEDDING")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "traceback": traceback.format_exc()},
            )
# ========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("❌ UNHANDLED EXCEPTION")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )
