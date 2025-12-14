from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

class EmbedRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "active", "model": "intfloat/multilingual-e5-small"}

@app.post("/embed")
async def embed(item: EmbedRequest):
    global model
    try:
        if model is None:
            print("⏳ Loading Model for the first time...")
            # threads=1 prevents the VPS from freezing
            model = TextEmbedding(model_name="intfloat/multilingual-e5-small", threads=1)
            print("✅ Model Loaded!")

        # Handle empty text to prevent crashes
        if not item.text or item.text.strip() == "":
            return {"vector": []}

        # Convert
        embeddings = list(model.embed([item.text]))
        return {"vector": embeddings[0].tolist()}

    except Exception as e:
        # This prints the REAL error to Coolify logs and returns it to Directus
        print(f"❌ ERROR: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
