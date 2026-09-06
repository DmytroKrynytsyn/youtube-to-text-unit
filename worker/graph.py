import asyncio
import os
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from prometheus_client import Counter, Histogram

from worker import queues
from worker.transcript import build_transcript_context

MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "2"))

youtube_tasks_total = Counter(
    "youtube_tasks_total",
    "Total youtube-to-text tasks completed",
    ["status"],
)

youtube_task_duration_seconds = Histogram(
    "youtube_task_duration_seconds",
    "End-to-end youtube-to-text task duration in seconds (fetch + essay)",
)

EXTERNAL_ESSAY_PROMPT_TEMPLATE = """Here is a raw, auto-generated YouTube video transcript. It has not been cleaned up and may
contain punctuation, casing, and ASR/caption errors, filler words, repeated words, and false starts.

Title (deduct from text)
URL: {url}

Raw transcript:
{transcript}

---

Your task is to extract all the information from this transcript and present it as a single coherent, structured,
story, in form of essay.

Deduct actual language from the raw script, use it to generate the result text.

Respond clearly, in simple words, avoid long/complex constructions
Respond in the same language as the transcript ({lang}).

Follow this structure strictly:

**Title**
A sharp, informative title that captures the core idea of the whole video.

**Introduction** (2-3 sentences)
Briefly state what the video is about and why it matters.

**Key Points**
Use clearly labeled sections or a numbered list.
Each point should be concise but complete — do not omit any important idea from the transcript.
Crystallize, do not generalize.

If the transcript mentions some list of things, like "top 5 of methods" or "10 tools for..." - make sure the list of the items is included in the answer.

**Main story (as much as needed to communiccate all the information)**

**Conclusion** (2-3 sentences)
Summarize the main takeaway and its significance.

**Source**
{url}

Rules:
- Do not invent anything not present in the transcript
- Do not pad with filler phrases
- Preserve all specific facts, numbers, names, and examples
- Total response must be dense, not verbose
- Use plain text formatting with ** for bold headers"""


class JobState(TypedDict):
    request_id: str
    chat_id: int
    url: str
    user_priority: int
    title: str
    lang: str
    retries: dict[str, int]
    essay_input: str | None
    essay: str | None
    error: str | None
    started_at: float


async def fetch_and_split(state: JobState) -> JobState:
    title, lang, transcript = await build_transcript_context(state["url"])

    queues.log("transcript_fetched", request_id=state["request_id"])

    await queues.publish_telegram_response(
        state["chat_id"], state["request_id"],
        result="transcript obtained, generating essay...",
        error=None,
    )

    return {
        **state,
        "title": title,
        "lang": lang,
        "retries": {},
        "essay_input": transcript,
    }


def _schedule_retry(coro):
    """Fire-and-forget: wait out a backoff, then redispatch, without blocking
    the current message's ack (the RabbitMQ channel runs prefetch_count=1,
    so a blocking sleep here would stall delivery of unrelated messages)."""
    async def _run():
        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
        await coro
    asyncio.create_task(_run())


async def _dispatch_essay(state: JobState):
    prompt = EXTERNAL_ESSAY_PROMPT_TEMPLATE.format(
        title=state["title"],
        url=state["url"],
        lang=state["lang"],
        transcript=state["essay_input"],
    )
    await queues.publish_llm_request(
        queues.LLM_REQUEST_QUEUE_EXTERNAL,
        prompt,
        correlation_id=f"{state['request_id']}:essay",
        request_id=state["request_id"],
        chat_id=state["chat_id"],
        stage="essay",
    )


async def dispatch_essay(state: JobState) -> JobState:
    await _dispatch_essay(state)
    return state


async def await_essay(state: JobState) -> JobState:
    reply = interrupt({"awaiting": "essay"})

    if reply.get("error"):
        attempts = state["retries"].get("essay", 0) + 1
        if attempts > MAX_RETRIES:
            queues.log("essay_failed", request_id=state["request_id"],
                       attempts=attempts, error=reply["error"])
            return {**state, "error": f"essay generation failed after retry: {reply['error']}"}
        queues.log("essay_retry", request_id=state["request_id"],
                   attempt=attempts, error=reply["error"])
        _schedule_retry(_dispatch_essay(state))
        return {**state, "retries": {**state["retries"], "essay": attempts}}

    return {**state, "essay": reply["result"]}


def essay_router(state: JobState) -> str:
    if state.get("error"):
        return "finalize"
    if state.get("essay") is None:
        return "await_essay"
    return "finalize"


async def finalize(state: JobState) -> JobState:
    youtube_task_duration_seconds.observe(time.time() - state["started_at"])
    youtube_tasks_total.labels(status="error" if state.get("error") else "success").inc()

    await queues.publish_telegram_response(
        state["chat_id"], state["request_id"],
        result=state.get("essay"), error=state.get("error"),
    )
    return state


def build_graph(checkpointer):
    g = StateGraph(JobState)
    g.add_node("fetch_and_split", fetch_and_split)
    g.add_node("dispatch_essay", dispatch_essay)
    g.add_node("await_essay", await_essay)
    g.add_node("finalize", finalize)

    g.add_edge(START, "fetch_and_split")
    g.add_edge("fetch_and_split", "dispatch_essay")
    g.add_edge("dispatch_essay", "await_essay")
    g.add_conditional_edges("await_essay", essay_router, ["await_essay", "finalize"])
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
