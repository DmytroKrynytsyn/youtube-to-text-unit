import os
import json
import aio_pika

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq.rabbitmq.svc.cluster.local/")

TASK_QUEUE = "youtube-to-text-task"
RESPONSE_QUEUE = "telegram-response-message"
LLM_REQUEST_QUEUE_SAI = "llm_requests_sai"
LLM_REQUEST_QUEUE_MAI = "llm_requests_mai"
LLM_RESPONSE_QUEUE = "llm_responses"

rabbitmq_channel: aio_pika.Channel = None


def log(event: str, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


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


async def publish_llm_request(queue_name: str, prompt: str, correlation_id: str, **passthrough):
    body = json.dumps({"prompt": prompt, **passthrough})

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=body.encode(),
            correlation_id=correlation_id,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=queue_name,
    )

    log("llm_request_published", queue=queue_name, correlation_id=correlation_id,
        prompt_len=len(prompt), **passthrough)
