import os
import json
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import quote

import aio_pika
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
import psycopg

from worker import queues
from worker.queues import log
from worker.graph import build_graph


class FilterHealthMetrics:
    def filter(self, record) -> bool:
        msg = record.getMessage()
        return "/health" not in msg and "/metrics" not in msg


logging.getLogger("uvicorn.access").addFilter(FilterHealthMetrics())

DATABASE_URL = os.getenv("DATABASE_URL")
DB_SCHEMA = os.getenv("DB_SCHEMA", "youtube_to_text_unit")

rabbitmq_connection: aio_pika.RobustConnection = None
checkpointer_cm = None
checkpointer: AsyncPostgresSaver = None
compiled_graph = None


async def on_youtube_task(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        log("youtube_task_received", queue=queues.TASK_QUEUE, correlation_id=message.correlation_id,
            size_bytes=len(message.body))
        body = json.loads(message.body)
        chat_id = body.get("chat_id")
        url = body.get("url")
        request_id = body.get("request_id") or str(uuid.uuid4())

        if not chat_id or not url:
            log("youtube_task_missing_fields", request_id=request_id)
            return

        initial_state = {
            "request_id": request_id,
            "chat_id": chat_id,
            "url": url,
            "title": "",
            "lang": "en",
            "chunks": [],
            "total_chunks": 0,
            "cleaned": {},
            "retries": {},
            "round": 0,
            "essay_input": None,
            "essay": None,
            "error": None,
            "started_at": time.time(),
        }

        try:
            await compiled_graph.ainvoke(
                initial_state, config={"configurable": {"thread_id": request_id}}
            )
            log("youtube_task_started", chat_id=chat_id, url=url, request_id=request_id)
        except ValueError as e:
            await queues.publish_telegram_response(chat_id, request_id, None, f"could not parse URL: {e}")
        except Exception as e:
            log("youtube_task_error", chat_id=chat_id, url=url, error=str(e), request_id=request_id)
            await queues.publish_telegram_response(
                chat_id, request_id, None, "failed to fetch transcript, please try again"
            )


async def on_llm_response(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        log("llm_response_received", queue=queues.LLM_RESPONSE_QUEUE, correlation_id=message.correlation_id,
            size_bytes=len(message.body))
        try:
            body = json.loads(message.body)
        except Exception as e:
            log("llm_response_invalid_body", error=str(e))
            return

        request_id = body.get("request_id")
        if not request_id:
            log("llm_response_missing_request_id")
            return

        try:
            result = await compiled_graph.ainvoke(
                Command(resume=body), config={"configurable": {"thread_id": request_id}}
            )
        except Exception as e:
            log("llm_response_resume_error", request_id=request_id, error=str(e))
            return

        if "__interrupt__" not in result:
            await checkpointer.adelete_thread(request_id)
            log("job_finalized", request_id=request_id)


async def setup_consumer():
    channel = await rabbitmq_connection.channel()
    queues.rabbitmq_channel = channel

    task_queue = await channel.declare_queue(queues.TASK_QUEUE, durable=True)
    await task_queue.consume(on_youtube_task)

    await channel.declare_queue(queues.RESPONSE_QUEUE, durable=True)
    await channel.declare_queue(queues.LLM_REQUEST_QUEUE_SAI, durable=True)
    await channel.declare_queue(queues.LLM_REQUEST_QUEUE_MAI, durable=True)

    await channel.set_qos(prefetch_count=1)
    llm_response_queue = await channel.declare_queue(queues.LLM_RESPONSE_QUEUE, durable=True)
    await llm_response_queue.consume(on_llm_response)

    log("consumer_registered", task_queue=queues.TASK_QUEUE, response_queue=queues.RESPONSE_QUEUE,
        llm_response_queue=queues.LLM_RESPONSE_QUEUE)


async def setup_schema():
    async with await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rabbitmq_connection, checkpointer_cm, checkpointer, compiled_graph

    await setup_schema()

    db_uri = f"{DATABASE_URL}?options={quote(f'-c search_path={DB_SCHEMA}')}"
    checkpointer_cm = AsyncPostgresSaver.from_conn_string(db_uri)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()
    compiled_graph = build_graph(checkpointer)

    rabbitmq_connection = await aio_pika.connect_robust(queues.RABBITMQ_URL)
    rabbitmq_connection.reconnect_callbacks.add(lambda *_: asyncio.create_task(setup_consumer()))

    await setup_consumer()

    log("startup", rabbitmq_url=queues.RABBITMQ_URL, task_queue=queues.TASK_QUEUE)

    yield

    await checkpointer_cm.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"healthy": True}
