from fastapi import FastAPI
from pydantic import BaseModel
from fastembed import TextEmbedding
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS so your React dashboard can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model (Downloads once on startup ~80MB)
# "intfloat/multilingual-e5-small" is excellent for Persian
model = TextEmbedding(model_name="intfloat/multilingual-e5-small")

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(item: EmbedRequest):
    # Convert text to vector
    embeddings = list(model.embed([item.text]))
    return {"vector": embeddings[0].tolist()}
