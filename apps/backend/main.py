# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import io
import uuid
from typing import List

import fitz
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
BUCKET = os.getenv("GCS_BUCKET", "")

app = FastAPI(title="LegalLight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_vertex():
    if not PROJECT_ID:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def extract_pdf_text(file_bytes: bytes) -> str:
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        texts = [p.get_text() for p in doc]
    return "\n".join(texts).strip()

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if len(current) + len(p) + 1 <= max_chars:
            current = current + ("\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

def embed(texts: List[str]) -> np.ndarray:
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    res = model.get_embeddings(texts)
    return np.array([e.values for e in res], dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

def summarize(text: str) -> str:
    model = GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are LegalLight. Rewrite the following legal text in plain language. "
        "Do not provide legal advice.\n\n"
        f"{text}\n\n"
        "Output: bullets for overview, obligations, rights, fees/dates, risks, and "
        "a <=200 word plain summary."
    )
    return model.generate_content(prompt).text

def answer(question: str, chunks: List[str]) -> str:
    model = GenerativeModel("gemini-1.5-flash")
    stitched = ""
    for i, ch in enumerate(chunks):
        stitched += f"[Chunk {i+1}]\n{ch}\n\n"
    prompt = (
        "Answer strictly using the provided context. If unknown, say you cannot find it. "
        "Avoid legal advice; explain in plain language. Quote short excerpts with chunk numbers.\n\n"
        f"Question:\n{question}\n\nContext:\n{stitched}\nAnswer:"
    )
    return model.generate_content(prompt).text

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload(file: UploadFile):
    init_vertex()
    if not BUCKET:
        raise HTTPException(500, "GCS_BUCKET not configured")
    data = await file.read()
    text = extract_pdf_text(data)

    # Store file in GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    doc_id = str(uuid.uuid4())
    blob = bucket.blob(f"uploads/{doc_id}.pdf")
    blob.upload_from_string(data, content_type=file.content_type)

    # Basic in-memory processing for demo (no DB)
    chunks = chunk_text(text)
    # Keep a simple per-process store (for hackathon demo)
    app.state.docs = getattr(app.state, "docs", {})
    app.state.docs[doc_id] = {"text": text, "chunks": chunks}
    return {"doc_id": doc_id, "num_chunks": len(chunks)}

@app.post("/api/doc/{doc_id}/summarize")
def api_summarize(doc_id: str):
    init_vertex()
    doc = getattr(app.state, "docs", {}).get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {"summary": summarize(doc["text"])}

@app.post("/api/doc/{doc_id}/ask")
def api_ask(doc_id: str, q: dict):
    init_vertex()
    doc = getattr(app.state, "docs", {}).get(doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    question = q.get("question", "")
    if not question.strip():
        raise HTTPException(400, "Missing question")
    # Retrieve top-5 similar chunks
    q_vec = embed([question])[0]
    c_vecs = embed(doc["chunks"])
    sims = [cosine_sim(q_vec, v) for v in c_vecs]
    top = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[:5]
    top_chunks = [doc["chunks"][i] for i, _ in top]
    return {"answer": answer(question, top_chunks)}
