from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model (starts empty)
model = None

class EmbedRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "active", "model": "intfloat/multilingual-e5-small"}

@app.post("/embed")
async def embed(item: EmbedRequest):
    global model
    # Lazy Load: Only download/load the model when actually needed
    if model is None:
        print("Model not loaded. Downloading/Loading now...")
        model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
        print("Model loaded successfully!")

    # Convert text to vector
    embeddings = list(model.embed([item.text]))
    return {"vector": embeddings[0].tolist()}
