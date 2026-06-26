import os
import time
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from prometheus_client import Counter, Histogram

from worker import queues
from worker.transcript import build_transcript_context

CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "480"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "50"))
MAX_RETRIES = 1

youtube_tasks_total = Counter(
    "youtube_tasks_total",
    "Total youtube-to-text tasks completed",
    ["status"],
)

youtube_task_duration_seconds = Histogram(
    "youtube_task_duration_seconds",
    "End-to-end youtube-to-text task duration in seconds (fetch + chunk cleanup + essay)",
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE_CHARS,
    chunk_overlap=CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)

CHUNK_PROMPT_TEMPLATE = """You are cleaning up one fragment of a raw YouTube auto-generated transcript.
This is fragment {chunk_index} of {total_chunks}, in order. You do not see the other fragments.

Video title: {title}
Language: {lang}

Raw fragment:
{chunk_text}

---

Task: rewrite this fragment as clean, dense, factual prose in {lang}.

Rules:
- Fix punctuation, casing, and obvious ASR/caption errors (repeated words, false starts, filler words like "um", "uh", "you know").
- Do not summarize away facts: keep every concrete fact, number, name, date, and example exactly as stated.
- Do not add an introduction, conclusion, title, or any commentary — output only the cleaned content of this fragment.
- Do not guess at content from other fragments — only use what's in this fragment.
- If the fragment starts or ends mid-sentence, that's expected (chunks overlap) — clean it up as best as you can without inventing missing words.
- Output plain prose, no bullet points, no headers, no markdown.
- Respond only in {lang}."""

ESSAY_PROMPT_TEMPLATE = """Here are {total_chunks} cleaned, ordered fragments of a YouTube video transcript. They were
cleaned up independently and may contain minor duplicate or overlapping phrases at their boundaries.

Title: {title}
URL: {url}
Language: {lang}

Cleaned fragments (in order):
{joined_segments}

---

Your task is to merge these fragments into a single coherent, structured, semi-academic one-page essay.
Respond clearly, in simple words, avoid long/complex constructions
Respond in the same language as the fragments ({lang}).

Follow this structure strictly:

**Title**
A sharp, informative title that captures the core idea of the whole video.

**Introduction** (2-3 sentences)
Briefly state what the video is about and why it matters.

**Key Points**
Use clearly labeled sections or a numbered list.
Each point should be concise but complete — do not omit any important idea from the fragments.
Crystallize, do not generalize.

If the fragments mention some list of things, like "top 5 of methods" or "10 tools for..." - make sure the list of the items is included in the answer.

**Conclusion** (2-3 sentences)
Summarize the main takeaway and its significance.

**Source**
{url}

Rules:
- Do not invent anything not present in the fragments
- Do not pad with filler phrases
- Preserve all specific facts, numbers, names, and examples
- If adjacent fragments repeat the same idea due to overlap, mention it only once
- Total response must be under 2000 characters — be dense, not verbose
- Use plain text formatting with ** for bold headers"""


class JobState(TypedDict):
    request_id: str
    chat_id: int
    url: str
    title: str
    lang: str
    chunks: list[str]
    total_chunks: int
    cleaned: dict[int, str]
    retries: dict[str, int]
    essay: str | None
    error: str | None
    started_at: float


async def fetch_and_split(state: JobState) -> JobState:
    title, lang, transcript = await build_transcript_context(state["url"])
    chunks = splitter.split_text(transcript)
    total = len(chunks)

    await queues.publish_telegram_response(
        state["chat_id"], state["request_id"],
        result=f"transcript obtained, split into {total} part(s) — cleaning up...",
        error=None,
    )

    for i, chunk in enumerate(chunks):
        await _dispatch_chunk(state["request_id"], state["chat_id"], title, lang, chunks, i, total)

    return {
        **state,
        "title": title,
        "lang": lang,
        "chunks": chunks,
        "total_chunks": total,
        "cleaned": {},
        "retries": {},
    }


async def _dispatch_chunk(request_id, chat_id, title, lang, chunks, chunk_index, total_chunks):
    prompt = CHUNK_PROMPT_TEMPLATE.format(
        chunk_index=chunk_index + 1,
        total_chunks=total_chunks,
        title=title,
        lang=lang,
        chunk_text=chunks[chunk_index],
    )
    await queues.publish_llm_request(
        queues.LLM_REQUEST_QUEUE_SAI,
        prompt,
        correlation_id=f"{request_id}:{chunk_index}",
        request_id=request_id,
        chat_id=chat_id,
        stage="chunk",
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )


async def await_chunk(state: JobState) -> JobState:
    reply = interrupt({"awaiting": "chunk", "have": len(state["cleaned"]), "total": state["total_chunks"]})
    idx = reply["chunk_index"]

    if reply.get("error"):
        key = str(idx)
        attempts = state["retries"].get(key, 0) + 1
        if attempts > MAX_RETRIES:
            queues.log("chunk_failed", request_id=state["request_id"], chunk_index=idx,
                       attempts=attempts, error=reply["error"])
            return {**state, "error": f"chunk {idx} failed after retry: {reply['error']}"}
        queues.log("chunk_retry", request_id=state["request_id"], chunk_index=idx,
                   attempt=attempts, error=reply["error"])
        await _dispatch_chunk(
            state["request_id"], state["chat_id"], state["title"], state["lang"],
            state["chunks"], idx, state["total_chunks"],
        )
        return {**state, "retries": {**state["retries"], key: attempts}}

    return {**state, "cleaned": {**state["cleaned"], idx: reply["result"]}}


def chunk_router(state: JobState) -> str:
    if state.get("error"):
        return "finalize"
    if len(state["cleaned"]) < state["total_chunks"]:
        return "await_chunk"
    return "dispatch_essay"


async def _dispatch_essay(state: JobState):
    ordered = [state["cleaned"][i] for i in range(state["total_chunks"])]
    joined = "\n\n".join(ordered)
    prompt = ESSAY_PROMPT_TEMPLATE.format(
        total_chunks=state["total_chunks"],
        title=state["title"],
        url=state["url"],
        lang=state["lang"],
        joined_segments=joined,
    )
    await queues.publish_llm_request(
        queues.LLM_REQUEST_QUEUE_MAI,
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
        await _dispatch_essay(state)
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
    g.add_node("await_chunk", await_chunk)
    g.add_node("dispatch_essay", dispatch_essay)
    g.add_node("await_essay", await_essay)
    g.add_node("finalize", finalize)

    g.add_edge(START, "fetch_and_split")
    g.add_edge("fetch_and_split", "await_chunk")
    g.add_conditional_edges("await_chunk", chunk_router, ["await_chunk", "dispatch_essay", "finalize"])
    g.add_edge("dispatch_essay", "await_essay")
    g.add_conditional_edges("await_essay", essay_router, ["await_essay", "finalize"])
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)
