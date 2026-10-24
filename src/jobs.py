import logging
from pathlib import Path

from redis import Redis
from rq import Queue, Retry
from rq.exceptions import NoSuchJobError
from rq.job import Job

from src.app import GraphRAG
from src.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

_cfg = Config()

redis_conn = Redis.from_url(_cfg.redis_url)
ingest_queue = Queue("ingest", connection=redis_conn)


def ingest_job(path: str) -> dict[str, int]:
    """Runs inside the worker process. The worker forks for every job, so open
    fresh connections here instead of reusing ones created before the fork."""
    rag = GraphRAG()
    try:
        return rag.ingest_pdf(path)
    finally:
        rag.close()


def enqueue_ingest(path: Path) -> str:
    """Put an ingest job on the queue and return its id."""
    job = ingest_queue.enqueue(
        ingest_job,
        str(path.resolve()),
        job_timeout="2h",
        result_ttl=86400,    # keep the result for 1 day
        failure_ttl=86400,   # keep failure info for 1 day
        # Retries are cheap: chunks that already succeeded are served from the LLM cache
        retry=Retry(max=2, interval=[30, 120]),
    )
    logger.info("Enqueued job %s for %s", job.id, path.name)
    return job.id


def get_job_info(job_id: str) -> dict | None:
    """Return the job status (queued / started / finished / failed), or None if unknown."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return None

    status = job.get_status()
    info: dict = {"status": status, "filename": Path(job.args[0]).name}
    if status == "finished":
        info["result"] = job.result
    elif status == "failed":
        info["error"] = job.exc_info
    return info