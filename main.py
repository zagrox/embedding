import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware

# 1. Force Cache Directory
CACHE_DIR = "/app/fastembed_cache"
os.environ["FASTEMBED_CACHE_PATH"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# Using a standard supported multilingual model
# This model outputs vector size: 384
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class EmbedRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "status": "active", 
        "model": MODEL_NAME,
        "cache_dir": CACHE_DIR,
        "model_loaded": model is not None
    }

@app.post("/embed")
async def embed(item: EmbedRequest):
    global model
    print(f"📥 Received request: {item.text[:50]}...")

    try:
        if model is None:
            print(f"⏳ Downloading model {MODEL_NAME} to {CACHE_DIR} ...")
            model = TextEmbedding(
                model_name=MODEL_NAME, 
                threads=1,
                cache_dir=CACHE_DIR
            )
            print("✅ Model Loaded Successfully!")

        print("🧮 Calculating Vector...")
        embeddings = list(model.embed([item.text]))
        vector = embeddings[0].tolist()
        
        print(f"✅ Success! Vector length: {len(vector)}")
        return {"vector": vector}

    except Exception as e:
        print("❌ CRITICAL ERROR ❌")
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content={"error": str(e), "type": type(e).__name__}
        )
