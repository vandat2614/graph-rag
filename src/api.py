import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.app import GraphRAG
from src.config import Config
from src.jobs import enqueue_ingest, get_job_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

cfg = Config()
UPLOAD_DIR = Path(cfg.data_dir) / "uploads"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app.state.rag = GraphRAG()  # used for /ask only; ingestion runs in the worker
    yield
    app.state.rag.close()


app = FastAPI(title="Simple Graph RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Node(BaseModel):
    id: str
    name: str
    type: str
    description: str


class Edge(BaseModel):
    id: str
    source: str  # node id
    target: str  # node id
    keywords: str
    description: str


class AskResponse(BaseModel):
    answer: str
    # The part of the knowledge graph that was used to answer, ready to be drawn
    nodes: list[Node]
    edges: list[Edge]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Demo page (single static HTML file)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", status_code=202)
def upload_document(file: UploadFile = File(...)) -> dict:
    """Upload a PDF. Indexing runs in a worker; poll /jobs/{job_id} for progress."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Path(...).name strips any directory part from the client-supplied name
    path = UPLOAD_DIR / Path(file.filename).name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"job_id": enqueue_ingest(path)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    info = get_job_info(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return info


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answer, ctx = app.state.rag.ask_with_context(req.question)
    return AskResponse(
        answer=answer,
        nodes=[
            Node(id=e.id, name=e.name, type=e.type, description=e.description)
            for e in ctx.entities
        ],
        edges=[
            Edge(
                id=r.id,
                source=r.source_id,
                target=r.target_id,
                keywords=r.keywords,
                description=r.description,
            )
            for r in ctx.relations
        ],
    )