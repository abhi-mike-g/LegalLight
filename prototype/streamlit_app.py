# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 AMG One

import io
import os
import re
import uuid
from typing import List, Tuple

import numpy as np
import streamlit as st
import vertexai
from sklearn.neighbors import NearestNeighbors
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

DISCLAIMER = (
    "LegalLight is an educational tool that simplifies legal language. "
    "It is not legal advice. Consult a qualified attorney for advice on your situation."
)

RISK_PATTERNS: List[Tuple[str, str, str]] = [
    ("High", "Arbitration / Class Action Waiver", r"\b(arbitration|class action waiver|waive.*jury)\b"),
    ("High", "Early Termination / Cancellation Fees", r"\b(early termination|break fee|cancellation fee|liquidated damages)\b"),
    ("High", "Limitation of Liability / Indemnity", r"\b(limitation of liability|indemnif(y|ication)|hold harmless)\b"),
    ("Medium", "Auto-Renewal", r"\b(auto[- ]?renew|automatic renewal|renewal term)\b"),
    ("Medium", "Unilateral Changes", r"\b(at our sole discretion|we may change|subject to change without notice)\b"),
    ("Medium", "Data Sharing / Third Parties", r"\b(share.*data|third[- ]?parties|sell.*data)\b"),
    ("Low", "Governing Law / Venue", r"\b(governing law|venue|jurisdiction)\b"),
    ("Low", "Payment Terms / Late Fees", r"\b(late fee|interest|finance charge|payment terms)\b"),
]

def init_vertex():
    if not PROJECT_ID:
        st.error("GOOGLE_CLOUD_PROJECT is not set. See setup instructions in README.")
        st.stop()
    vertexai.init(project=PROJECT_ID, location=LOCATION)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if fitz is None:
        st.error("PyMuPDF not installed. Please install 'pymupdf'.")
        st.stop()
    text = []
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text).strip()

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
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

def embed_texts(texts: List[str]) -> np.ndarray:
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    # API accepts batches; here we call once per up to 100 texts for simplicity
    vectors = []
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 100:
            res = model.get_embeddings(batch)
            vectors.extend([e.values for e in res])
            batch = []
    if batch:
        res = model.get_embeddings(batch)
        vectors.extend([e.values for e in res])
    return np.array(vectors, dtype=np.float32)

def build_retriever(chunks: List[str]):
    vectors = embed_texts(chunks)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    normed = vectors / norms
    nn = NearestNeighbors(n_neighbors=min(5, len(chunks)), metric="cosine")
    nn.fit(normed)
    return {"chunks": chunks, "vectors": normed, "nn": nn}

def retrieve_context(retriever, query: str, k: int = 5) -> List[Tuple[int, str, float]]:
    q_vec = embed_texts([query])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    distances, indices = retriever["nn"].kneighbors([q_vec], n_neighbors=min(k, len(retriever["chunks"])))
    pairs = []
    for dist, idx in zip(distances[0], indices[0]):
        sim = 1.0 - float(dist)
        pairs.append((int(idx), retriever["chunks"][idx], sim))
    return pairs

def simplify_text(full_text: str) -> str:
    model = GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"{DISCLAIMER}\n"
        "You are LegalLight. Rewrite the following legal text in plain, simple language "
        "that a 12th-grade reader can understand. Keep it accurate, avoid legal jargon, "
        "and include bullet points and examples where helpful. Do not give legal advice.\n\n"
        f"Text:\n{full_text}\n\n"
        "Output format:\n- Short overview (3–5 bullets)\n- Key obligations\n- Key rights\n- "
        "Important dates/fees/penalties\n- Risks to watch out for\n- Plain summary (<= 200 words)"
    )
    res = model.generate_content(prompt)
    return res.text

def answer_question(question: str, context_chunks: List[Tuple[int, str, float]]) -> str:
    model = GenerativeModel("gemini-1.5-flash")
    stitched = ""
    for i, (idx, ch, sim) in enumerate(context_chunks):
        stitched += f"[Chunk {i+1} | Sim {sim:.2f}]\n{ch}\n\n"
    prompt = (
        f"{DISCLAIMER}\n"
        "Answer the user's question using ONLY the context provided. "
        "If the answer is not in the context, say you cannot find it. "
        "Cite relevant phrasing by quoting short excerpts and indicate the chunk number. "
        "Do not provide legal advice; explain in plain language.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{stitched}\n"
        "Answer:"
    )
    res = model.generate_content(prompt)
    return res.text

def detect_risks(text: str) -> List[Tuple[str, str, List[str]]]:
    flags = []
    for severity, label, pattern in RISK_PATTERNS:
        hits = re.findall(pattern, text, flags=re.IGNORECASE)
        if hits:
            flags.append((severity, label, list(set(hits))))
    return flags

def main():
    st.set_page_config(page_title="LegalLight (AGPL-3.0)", page_icon="💡", layout="wide")
    st.title("LegalLight — Demystify Legal Documents (AGPL-3.0)")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("Settings")
        project = st.text_input("Google Cloud Project ID", value=PROJECT_ID)
        region = st.selectbox("Region", ["us-central1", "us-east1", "europe-west1"], index=0)
        if st.button("Apply"):
            os.environ["GOOGLE_CLOUD_PROJECT"] = project
            os.environ["GOOGLE_CLOUD_LOCATION"] = region
            st.experimental_rerun()

    init_vertex()

    uploaded = st.file_uploader("Upload a PDF contract", type=["pdf"])
    if uploaded is None:
        st.info("Upload a sample contract (PDF) to begin.")
        return

    file_bytes = uploaded.read()
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(file_bytes)

    st.subheader("Original Text (first 1000 chars)")
    st.text_area("Document Text", value=text[:1000] + ("..." if len(text) > 1000 else ""), height=200)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simplify Document"):
            with st.spinner("Generating simplified summary..."):
                summary = simplify_text(text)
            st.subheader("Simplified Summary")
            st.write(summary)

    with col2:
        chunks = chunk_text(text, max_chars=1500)
        retriever = build_retriever(chunks)
        question = st.text_input("Ask a question about this document", value="What happens if I end the lease early?")
        if st.button("Get Answer"):
            with st.spinner("Retrieving relevant clauses..."):
                ctx = retrieve_context(retriever, question, k=5)
                answer = answer_question(question, ctx)
            st.subheader("Answer (context-aware)")
            st.write(answer)
            with st.expander("Show retrieved chunks"):
                for i, (idx, ch, sim) in enumerate(ctx):
                    st.markdown(f"Chunk {i+1} (similarity {sim:.2f})")
                    st.text(ch)

    with st.expander("Risk Flags"):
        risks = detect_risks(text)
        if not risks:
            st.write("No obvious risk patterns detected by simple heuristics.")
        else:
            for severity, label, hits in risks:
                st.write(f"{severity} — {label} (matches: {', '.join(hits)})")

    st.divider()
    st.caption("Privacy: This prototype processes documents in-memory. No files are saved unless you modify the code to store them. Vertex AI terms apply.")

if __name__ == "__main__":
    main()
