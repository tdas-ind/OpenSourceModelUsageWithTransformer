from __future__ import annotations
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from contextlib import asynccontextmanager


MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/SFR-Embedding-Mistral"
BATCH_SIZE = 16

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str = "local-embedding"

class EmbeddingResponse(BaseModel):
    data: List[List[float]]
    model: str
    object: str = "list"

@asynccontextmanager
async def load_model(app: FastAPI):
    """Loads the model during FastAPI startup"""
    global MODEL, TOKENIZER
    try:
        print("Loading model...")
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        
        model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        
        if torch.cuda.is_available():
            model = torch.compile(model)
            model = model.half()  # FP16
        
        MODEL = model
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Error loading model: {str(e)}")
    yield

app = FastAPI(lifespan=load_model)

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last token's representation based on padding."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    
    try:
        max_length = 4096
        embeddings_list = []
        
        dataloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        
        for batch in dataloader:
            inputs = TOKENIZER(batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = MODEL(**inputs)
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            embeddings_list.extend(embeddings.cpu().tolist())

        return embeddings_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings_api(request: EmbeddingRequest):
    
    if not MODEL or not TOKENIZER:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        texts = [request.input] if isinstance(request.input, str) else request.input
        print("Processing texts:", texts)

        embeddings = get_embeddings(texts)

        response_data = {
            "data": embeddings, 
            "model": MODEL_NAME,
            "object": "list"
        }
        return response_data

    except Exception as e:
        print("ERROR:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "testApi:app",
        host="0.0.0.0",
        port=8002,
        loop="asyncio",
        log_level="debug"
    )
