import os, asyncio, time
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from chromadb.config import Settings

from .discover import discover_links
from .indexer import reindex, ensure_db

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

LINKTREE_URL = os.getenv("LINKTREE_URL", "https://linktr.ee/newhopechurchcc")
SYNC_TOKEN = os.getenv("SYNC_TOKEN", "")
SYNC_INTERVAL_HOURS = float(os.getenv("SYNC_INTERVAL_HOURS", "1"))
PORT = int(os.getenv("PORT", "10000"))

client = OpenAI()

# Vector DB (Chroma) persisted on disk
DB_DIR = "data/chroma"
os.makedirs(DB_DIR, exist_ok=True)
chroma = chromadb.Client(Settings(persist_directory=DB_DIR))
collection = chroma.get_or_create_collection(name="nhc")

app = FastAPI(title="New Hope Linktree Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Static + simple page
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class AskBody(BaseModel):
    question: str
    k: int = 6

def _auth_or_403(auth_header: Optional[str]):
    if not SYNC_TOKEN:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization")
    token = auth_header.split(" ", 1)[1].strip()
    if token != SYNC_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("app/templates/index.html")

@app.post("/sync")
def sync(authorization: Optional[str] = Header(None)):
    _auth_or_403(authorization)
    urls = discover_links(LINKTREE_URL)
    result = reindex(urls, collection)
    return {"discovered_urls": urls, **result}

@app.post("/ask")
def ask(body: AskBody):
    # embed question
    emb = client.embeddings.create(model="text-embedding-3-large", input=[body.question])
    qvec = emb.data[0].embedding
    # query
    res = collection.query(query_embeddings=[qvec], n_results=body.k, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    if not docs:
        return {"answer": "I don’t have that yet. Try again after the next refresh.", "sources": []}

    # Build context snippets for the LLM
    ctxs = []
    for doc, meta in zip(docs, metas):
        title = meta.get("title") or meta.get("url")
        url = meta.get("url")
        ctxs.append(f"[{title}]({url})\n{doc}")

    system = (
        "You are a helpful assistant for New Hope Church. "
        "Only answer using the provided context snippets. "
        "If the answer isn't in the snippets, say you don’t know. "
        "Always end with a short list of the most relevant source links."
    )
    prompt = (
        f"{system}\n\nQuestion: {body.question}\n\n"
        "Context snippets:\n" + "\n\n---\n\n".join(ctxs) +
        "\n\nAnswer briefly, then list the most relevant sources as bullet links."
    )

    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer = chat.choices[0].message.content
    return {"answer": answer, "sources": [m.get("url") for m in metas if m.get("url")]}

# ---------- periodic sync loop ----------
async def periodic_sync():
    # initial sleep to let service boot
    await asyncio.sleep(5)
    while True:
        try:
            urls = discover_links(LINKTREE_URL)
            reindex(urls, collection)
        except Exception as e:
            print("Sync error:", e)
        await asyncio.sleep(int(SYNC_INTERVAL_HOURS * 3600))

@app.on_event("startup")
async def on_startup():
    ensure_db()
    asyncio.create_task(periodic_sync())
