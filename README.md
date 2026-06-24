# youtube-to-text-unit

RabbitMQ worker running on k3s, backed by a local Ollama LLM. Sibling service to [`telegram-bot-on-llm`](https://github.com/DmytroKrynytsyn/telegram-bot-on-llm).

git push -> Github Action -> Docker Hub -> ArgoCD -> k3s

## How it works

This service owns the full "YouTube link -> formatted text" pipeline. It never talks to Telegram directly.

1. Consumes `{chat_id, url, request_id}` from the `youtube-to-text-task` queue.
2. Fetches video info and captions via `yt-dlp`, builds a formatting prompt.
3. Publishes the prompt to the shared `llm_requests` queue (same LLM broker `telegram-bot-on-llm` uses), with `reply_to` set to its own `llm_responses_youtube_to_text_unit` queue.
4. On the LLM's reply, publishes `{chat_id, result, error, request_id}` to the `telegram-response-message` queue, which `telegram-bot-on-llm` consumes and relays to the user as-is.

## Stack

`k3s` · `ArgoCD` · `Helm` · `GitHub Actions` · `Docker Hub` · `FastAPI` · `uv` · `aio-pika` · `yt-dlp`

## Bootstrap

```bash
# register the app with ArgoCD (no secret to create - this service holds no credentials)
kubectl apply -f https://raw.githubusercontent.com/DmytroKrynytsyn/youtube-to-text-unit/main/argocd/application.yaml
```
