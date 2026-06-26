# youtube-to-text-unit

RabbitMQ worker running on k3s, fanning transcript cleanup out across the [`llm-unit`](https://github.com/DmytroKrynytsyn/llm-unit) inference pools. Sibling service to [`telegram-bot-on-llm`](https://github.com/DmytroKrynytsyn/telegram-bot-on-llm).

git push -> Github Action -> Docker Hub -> ArgoCD -> k3s

## How it works

This service owns the full "YouTube link -> formatted essay" pipeline. It never talks to Telegram directly. The orchestration is a [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` (`worker/graph.py`), checkpointed to Postgres so an in-flight job survives a pod restart.

1. Consumes `{chat_id, url, request_id}` from the `youtube-to-text-task` queue.
2. Fetches video info and captions via `yt-dlp`, splits the transcript into overlapping chunks (`langchain-text-splitters`), and publishes an intermediate `{chat_id, result, request_id}` ("split into N parts") to `telegram-response-message`.
3. Publishes one cleanup request per chunk to `llm_requests_sai`, each carrying `{prompt, request_id, chat_id, stage:"chunk", chunk_index, total_chunks}`.
4. Consumes `llm_responses` (the single shared reply queue `llm-unit` always publishes to). As chunk replies arrive, in any order, they're recorded against the job's checkpoint. A failed chunk is retried once before the whole job aborts.
5. Once every chunk has a reply, joins them in order and publishes one essay-synthesis request to `llm_requests_mai` with `stage:"essay"`.
6. On the essay reply, publishes `{chat_id, result, error, request_id}` to `telegram-response-message` (which `telegram-bot-on-llm` relays to the user as-is) and deletes the job's checkpoint rows from Postgres.

## Stack

`k3s` · `ArgoCD` · `Helm` · `GitHub Actions` · `Docker Hub` · `FastAPI` · `uv` · `aio-pika` · `yt-dlp` · `LangGraph` · `LangChain` · `PostgreSQL`

## Bootstrap

```bash
# create the Postgres connection secret before ArgoCD syncs (uses the infra-postgres credentials)
kubectl create secret generic youtube-to-text-unit-secret \
  --from-literal=DATABASE_URL=postgresql://app:changeme@postgres.postgres.svc.cluster.local:5432/app \
  -n youtube-to-text-unit

# register the app with ArgoCD
kubectl apply -f https://raw.githubusercontent.com/DmytroKrynytsyn/youtube-to-text-unit/main/argocd/application.yaml
```
