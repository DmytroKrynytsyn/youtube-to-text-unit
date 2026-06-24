
import os
import re
import json
import uuid
import asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import aio_pika
import logging
from urllib.parse import urlparse, parse_qs
import urllib.request
import yt_dlp


class FilterHealthMetrics:
    def filter(self, record) -> bool:
        msg = record.getMessage()
        return "/health" not in msg and "/metrics" not in msg


logging.getLogger("uvicorn.access").addFilter(FilterHealthMetrics())

app = FastAPI()
Instrumentator().instrument(app).expose(app)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq.rabbitmq.svc.cluster.local/")
TASK_QUEUE = "youtube-to-text-task"
RESPONSE_QUEUE = "telegram-response-message"
LLM_REQUEST_QUEUE = "llm_requests"
LLM_REPLY_QUEUE = "llm_responses_youtube_to_text_unit"

YOUTUBE_PROMPT_TEMPLATE = """Here is a transcript from a YouTube video.

Title: {title}
URL: {url}
Language: {lang}

Transcript:
{transcript}

---

Your task is to transform this transcript into a structured, semi-academic one-page essay.
Respond clearly, in simple words, avoid long/complex constructions
Respond in the same language as the transcript ({lang}).

Follow this structure strictly:

**Title**
A sharp, informative title that captures the core idea of whole transcript.

**Introduction** (2-3 sentences)
Briefly state what the video is about and why it matters.

**Key Points**
Use clearly labeled sections or a numbered list.
Each point should be concise but complete — do not omit any important idea from the transcript.
Crystallize, do not generalize.

If transcript/title mentions some list of things, like "top 5 of methods" or "10 tools for..." - make sure the list of the items is included in the answer.

**Conclusion** (2-3 sentences)
Summarize the main takeaway and its significance.

**Source**
{url}

Rules:
- Do not invent anything not present in the transcript
- Do not pad with filler phrases
- Preserve all specific facts, numbers, names, and examples
- Total response must be under 2000 characters — be dense, not verbose
- Use plain text formatting with ** for bold headers"""

rabbitmq_connection: aio_pika.RobustConnection = None
rabbitmq_channel: aio_pika.Channel = None


def log(event: str, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_video_info(url: str) -> tuple[str, str]:
    """Returns (title, language_code)."""
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "no_warnings": True,
            "ignore_errors": True,
            "extract_flat": "in_playlist",
            "format": None,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                log("yt_dlp_no_info", url=url)
                return "Unknown Title", "en"
            title = info.get("title") or "Unknown Title"
            lang = info.get("language") or info.get("default_audio_language") or "en"
            lang = lang.split("-")[0].lower()
            log("yt_dlp_info_fetched", url=url, title=title, lang=lang)
            return title, lang
    except Exception as e:
        log("yt_dlp_error", url=url, error=str(e))
        return "Unknown Title", "en"


def fetch_transcript(video_id: str, lang: str) -> str:
    log("transcript_fetch_start", video_id=video_id, lang=lang)
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        # Match preferred language variants dynamically using wildcards
        ydl_opts = {
            "skip_download": True,
            "subtitleslangs": ["orig", f"{lang}.*", "ru.*", "en.*", ".*-orig", ".*"],
            "quiet": True,
            "no_warnings": True,
            "ignore_errors": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise RuntimeError(f"Failed to extract info via yt-dlp for video {video_id}")

        auto = info.get("automatic_captions", {})
        subtitles = info.get("subtitles", {})

        # Priority fallback check to find any valid native language track matching our target
        caps = None
        for key in ["orig", f"{lang}-orig", lang, "ru-orig", "ru", "en-orig", "en"]:
            if key in subtitles:
                caps = subtitles[key]
                break
            if key in auto:
                caps = auto[key]
                break

        # Fall back to any 'orig' string match if specific codes were missing
        if not caps:
            all_tracks = {**auto, **subtitles}
            orig_key = next((k for k in all_tracks.keys() if "orig" in k), None)
            if orig_key:
                caps = all_tracks[orig_key]
            else:
                caps = next(iter(subtitles.values()), None) or next(iter(auto.values()), None)

        if not caps:
            raise RuntimeError(f"No captions found for video {video_id}")

        cap_url = next((f["url"] for f in caps if f.get("ext") == "json3"), None)
        if not cap_url:
            raise RuntimeError(f"No json3 caption format for video {video_id}")

        req = urllib.request.Request(
            cap_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )

        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())

        text = " ".join(
            seg.get("utf8", "") for e in data.get("events", []) for seg in e.get("segs", []) if seg.get("utf8")
        ).replace("\n", " ")

        if not text.strip():
            raise RuntimeError(f"Extracted json3 text data payload evaluated as empty for {video_id}")

        log("transcript_fetch_success", video_id=video_id, chars=len(text))
        return text
    except RuntimeError:
        raise
    except Exception as e:
        log("transcript_error", video_id=video_id, error=str(e))
        raise RuntimeError(f"Could not fetch transcript for video {video_id}: {e}")


async def build_prompt(url: str) -> tuple[str, str]:
    loop = asyncio.get_event_loop()

    video_id = extract_video_id(url)
    title, lang = await loop.run_in_executor(None, fetch_video_info, url)
    transcript = await loop.run_in_executor(None, fetch_transcript, video_id, lang)

    prompt = YOUTUBE_PROMPT_TEMPLATE.format(
        title=title,
        url=url,
        transcript=transcript,
        lang=lang,
    )
    return prompt, title


async def publish_telegram_response(chat_id: int, request_id: str, result: str | None, error: str | None):
    body = json.dumps({
        "chat_id": chat_id,
        "result": result,
        "error": error,
        "request_id": request_id,
    })

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=body.encode(),
            correlation_id=request_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=RESPONSE_QUEUE,
    )

    log("telegram_response_published", request_id=request_id, chat_id=chat_id, has_error=bool(error))


async def publish_llm_request(chat_id: int, request_id: str, prompt: str):
    body = json.dumps({
        "prompt": prompt,
        "request_id": request_id,
        "chat_id": chat_id,
    })

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=body.encode(),
            correlation_id=request_id,
            reply_to=LLM_REPLY_QUEUE,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=LLM_REQUEST_QUEUE,
    )

    log("llm_request_published", request_id=request_id, chat_id=chat_id, prompt_len=len(prompt))


async def on_youtube_task(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        body = json.loads(message.body)
        chat_id = body.get("chat_id")
        url = body.get("url")
        request_id = body.get("request_id") or str(uuid.uuid4())

        if not chat_id or not url:
            log("youtube_task_missing_fields", request_id=request_id)
            return

        try:
            prompt, title = await build_prompt(url)
            log("youtube_transcript_fetched", chat_id=chat_id, url=url, title=title, request_id=request_id)
            await publish_llm_request(chat_id, request_id, prompt)
        except ValueError as e:
            await publish_telegram_response(chat_id, request_id, None, f"could not parse URL: {e}")
        except Exception as e:
            log("youtube_task_error", chat_id=chat_id, url=url, error=str(e), request_id=request_id)
            await publish_telegram_response(chat_id, request_id, None, "failed to fetch transcript, please try again")


async def on_llm_response(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        try:
            body = json.loads(message.body)
            chat_id = body.get("chat_id")
            result = body.get("result")
            error = body.get("error")
            request_id = body.get("request_id")

            if not chat_id:
                log("llm_response_missing_chat_id", request_id=request_id)
                return

            await publish_telegram_response(chat_id, request_id, result, error)
        except Exception as e:
            log("llm_response_handler_error", error=str(e))


async def setup_consumer():
    global rabbitmq_channel
    rabbitmq_channel = await rabbitmq_connection.channel()

    task_queue = await rabbitmq_channel.declare_queue(TASK_QUEUE, durable=True)
    await task_queue.consume(on_youtube_task)

    await rabbitmq_channel.declare_queue(RESPONSE_QUEUE, durable=True)

    await rabbitmq_channel.declare_queue(LLM_REQUEST_QUEUE, durable=True)
    llm_reply_queue = await rabbitmq_channel.declare_queue(LLM_REPLY_QUEUE, durable=True)
    await llm_reply_queue.consume(on_llm_response)

    log("consumer_registered", task_queue=TASK_QUEUE, response_queue=RESPONSE_QUEUE, llm_reply_queue=LLM_REPLY_QUEUE)


@app.get("/health")
def health():
    return {"healthy": True}


@app.on_event("startup")
async def startup():
    global rabbitmq_connection

    rabbitmq_connection = await aio_pika.connect_robust(RABBITMQ_URL)
    rabbitmq_connection.reconnect_callbacks.add(lambda *_: asyncio.create_task(setup_consumer()))

    await setup_consumer()

    log("startup", rabbitmq_url=RABBITMQ_URL, task_queue=TASK_QUEUE)
