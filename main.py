from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow connections from anywhere (simplifies Coolify setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model (Downloads on first run)
# Using 'multilingual-e5-small' for Persian support
print("Loading model...")
model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
print("Model loaded!")

class EmbedRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "active", "model": "intfloat/multilingual-e5-small"}

@app.post("/embed")
async def embed(item: EmbedRequest):
    # Convert text to vector
    embeddings = list(model.embed([item.text]))
    return {"vector": embeddings[0].tolist()}
