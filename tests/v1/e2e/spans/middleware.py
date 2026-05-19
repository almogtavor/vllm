"""
Middleware for OpenAI-compatible API with modular processing.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

import httpx
from fastapi import FastAPI, Request
from openai import OpenAI
from pydantic import BaseModel, Field

from config_models import MiddlewareConfig
from processors import PromptProcessor
from logger import MiddlewareLogger


# Request Models
class ChatMessage(BaseModel):
    """Model for a single chat message."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message author"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """Model for chat completion requests."""

    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(
        ..., description="List of messages comprising the conversation"
    )
    max_tokens: Optional[int] = Field(
        default=128, description="Maximum number of tokens to generate", ge=1
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature between 0 and 2", ge=0.0, le=2.0
    )
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream back partial progress"
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for deterministic generation"
    )


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_stream(
    stream, start_time: float, request_id: str, middleware_logger
) -> tuple[str, Optional[str], float]:
    """
    Process streaming response from the backend.

    Args:
        stream: The streaming response from OpenAI client
        start_time: Time when the request started
        request_id: Unique request identifier
        middleware_logger: Logger instance for metrics

    Returns:
        Tuple of (full_text, finish_reason, ttft_seconds)
    """
    full_text = ""
    finish_reason = None
    ttft_measured = False
    ttft_seconds = 0.0

    for chunk in stream:
        if not ttft_measured:
            ttft_seconds = time.time() - start_time
            middleware_logger.log_ttft(request_id, ttft_seconds)
            ttft_measured = True

        if chunk.choices:
            choice = chunk.choices[0]
            # /v1/completions chunks have `text`; /v1/chat/completions chunks
            # have `delta.content` instead. Handle both shapes.
            text_piece = getattr(choice, "text", None)
            if text_piece is None:
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text_piece = getattr(delta, "content", None)
            if text_piece:
                full_text += text_piece
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    return full_text, finish_reason, ttft_seconds


def calculate_usage_info(
    full_prompt: List[int], full_text: str, ttft_seconds: float, tokenizer
) -> Dict[str, Any]:
    """
    Calculate token usage information.

    Args:
        full_prompt: Tokenized prompt
        full_text: Generated text
        ttft_seconds: Time to first token
        tokenizer: Tokenizer instance
    """
    output_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    return {
        "prompt_tokens": len(full_prompt),
        "completion_tokens": len(output_tokens),
        "total_tokens": len(full_prompt) + len(output_tokens),
        "ttft": ttft_seconds,
    }


def trim_response(text: str, trim_start: Optional[str], trim_finish: Optional[str]) -> str:
    """
    Trim response text by locating trim_finish first, then the nearest preceding trim_start.
    If both markers are found, remove the content between and including them.
    If only trim_finish is found, remove everything from the start through trim_finish.
    
    Args:
        text: The text to trim
        trim_start: String marking the start of content to remove (inclusive)
        trim_finish: String marking the end of content to remove (inclusive)
    
    Returns:
        Text with the matched section removed, or original text if trim_finish is not found
    """
    if not trim_finish:
        return text

    finish_start_idx = text.find(trim_finish)
    if finish_start_idx == -1:
        return text

    finish_end_idx = finish_start_idx + len(trim_finish)

    start_idx = -1
    if trim_start:
        start_idx = text.rfind(trim_start, 0, finish_start_idx)

    if start_idx != -1:
        trimmed_text = text[:start_idx] + text[finish_end_idx:]
    else:
        trimmed_text = text[finish_end_idx:]

    return trimmed_text.strip('\n')


def build_response(
    request_id: str,
    full_text: str,
    finish_reason: Optional[str],
    usage_info: Optional[dict],
) -> Dict[str, Any]:
    """
    Build the final response object.

    Args:
        request_id: Unique request identifier
        full_text: Generated text
        finish_reason: Reason for completion (can be None)

    Returns:
        Response dictionary in OpenAI format
    """
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "usage": usage_info,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": finish_reason,
            }
        ],
    }


def create_app(
    config_path: str | None = None,
) -> FastAPI:
    """Create FastAPI application with middleware."""

    # Load configuration - prefer env var, then argument, then default
    config_path = (
        config_path
        or os.environ.get("MIDDLEWARE_CONFIG_PATH")
        or "middleware_config.yaml"
    )
    config = MiddlewareConfig.from_yaml(config_path)

    # Apply env var overrides for backend config
    if "VLLM_ADDRESS" in os.environ:
        config.backend.base_url = os.environ["VLLM_ADDRESS"]
    if "MODEL_NAME" in os.environ:
        config.backend.model = os.environ["MODEL_NAME"]
        config.model.model_id = os.environ["MODEL_NAME"]

    # Validate and sync model configuration
    if not config.backend.model and not config.model.model_id:
        logger.error(
            "Model not defined. Please set MODEL_NAME environment variable or "
            "configure backend.model or model.model_id in the config file."
        )
        raise ValueError(
            "Model not defined. Please set MODEL_NAME environment variable or "
            "configure backend.model or model.model_id in the config file."
        )
    elif config.backend.model and not config.model.model_id:
        config.model.model_id = config.backend.model
    elif config.model.model_id and not config.backend.model:
        config.backend.model = config.model.model_id

    # Apply env var overrides for processing config
    _bool_env = lambda k: os.environ[k].lower() in (
        "true",
        "1",
        "yes",
    )
    if "MIDDLEWARE_PADDING_ENABLED" in os.environ:
        config.processing.padding.enabled = _bool_env(
            "MIDDLEWARE_PADDING_ENABLED",
        )
    if "MIDDLEWARE_DELIMITER_SPLITTING_ENABLED" in os.environ:
        config.processing.delimiter_splitting.enabled = _bool_env(
            "MIDDLEWARE_DELIMITER_SPLITTING_ENABLED"
        )
    if "SPAN_MODE" in os.environ:
        mode = os.environ["SPAN_MODE"]
        config.processing.span_mode.mode = mode
        config.processing.span_mode.enabled = mode in (
            "naive",
            "spans",
        ) or mode.startswith("legolink-")
    if "MIDDLEWARE_WARMUP_ENABLED" in os.environ:
        config.processing.warmup.enabled = _bool_env(
            "MIDDLEWARE_WARMUP_ENABLED",
        )

    # Apply env var overrides for span token IDs
    if "VLLM_V1_SPANS_TOKEN_PLUS" in os.environ:
        config.processing.span_mode.plus_token_id = int(
            os.environ["VLLM_V1_SPANS_TOKEN_PLUS"]
        )
    if "VLLM_V1_SPANS_TOKEN_CROSS" in os.environ:
        config.processing.span_mode.recompute_token_id = int(
            os.environ["VLLM_V1_SPANS_TOKEN_CROSS"]
        )
    if "VLLM_V1_SPANS_PAD_TOKEN" in os.environ:
        config.processing.padding.pad_token_id = int(
            os.environ["VLLM_V1_SPANS_PAD_TOKEN"]
        )

    app = FastAPI(title="Middleware")

    # Create httpx client with SSL verification disabled (like curl -k)
    http_client = httpx.Client(verify=False)

    # Create OpenAI client with validated config and custom http client
    openai_client = OpenAI(
        base_url=config.backend.base_url,
        api_key=config.backend.api_key,
        timeout=config.backend.timeout,
        http_client=http_client,
    )

    # Initialize processor and logger
    processor = PromptProcessor(config, openai_client)
    middleware_logger = MiddlewareLogger(config)

    # Store in app state
    app.state.config = config
    app.state.tokenizer = processor.tokenizer
    app.state.client = openai_client
    app.state.processor = processor
    app.state.logger = middleware_logger

    @app.on_event("startup")
    async def startup_event():
        """Log startup information."""
        logger.info("Middleware started with config: %s", config_path)
        logger.info("Backend URL: %s", config.backend.base_url)
        logger.info("Model: %s", config.model.model_id)

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model.model_id,
                    "object": "model",
                }
            ],
        }

    @app.get("/v1/metrics")
    async def get_metrics():
        """Get performance metrics."""
        return app.state.logger.metrics.get_metrics()

    @app.delete("/v1/metrics")
    async def clear_metrics():
        """Clear performance metrics."""
        count = app.state.logger.metrics.clear_metrics()
        return {"message": f"Cleared {count} metrics"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Handle chat completion requests."""
        model = request.model
        messages = [msg.model_dump() for msg in request.messages]
        max_tokens = request.max_tokens
        temperature = request.temperature
        seed = request.seed

        # Generate request ID
        request_id = app.state.logger.get_next_request_id()
        app.state.logger.log_request_start(request_id)

        logger.debug(
            "Request %s: Processing prompt with %d messages", request_id, len(messages)
        )

        # Process prompt through pipeline
        full_prompt, span_starts, cross_span_starts = app.state.processor.process_prompt(
            messages, model, max_tokens, temperature, seed
        )

        logger.debug(
            "Request %s: Prompt processed, %d tokens", request_id, len(full_prompt)
        )

        # Use backend model if configured, otherwise use request model
        backend_model = config.backend.model if config.backend.model else model

        # Main inference with streaming
        start_time = time.time()
        try:
            # Build completion parameters
            completion_params = {
                "model": backend_model,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            }
            # Add seed if provided
            if seed is not None:
                completion_params["seed"] = seed
            
            # Send-positions path: ship the *text* form of the pre-tokenised
            # prompt via /v1/chat/completions with a no-op chat_template so vLLM
            # tokenises it back to the same id list. /v1/chat/completions
            # vllm_xargs accepts list values, unlike /v1/completions.
            # Requires the prompt's PAD token to be a Llama-3 reserved special
            # token (e.g. 128002) so trailing pads survive the BPE round-trip.
            if (config.processing.span_mode.send_pos_not_delim and
                span_starts is not None and
                cross_span_starts is not None):
                prompt_text = app.state.tokenizer.decode(
                    full_prompt, skip_special_tokens=False
                )
                chat_params = {
                    "model": backend_model,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "extra_body": {
                        "chat_template": "{{- messages[0].content -}}",
                        "add_generation_prompt": False,
                        "add_special_tokens": False,
                        "vllm_xargs": {
                            "span_starts": span_starts,
                            "cross_span_starts": cross_span_starts,
                        },
                    },
                }
                if seed is not None:
                    chat_params["seed"] = seed
                logger.info(
                    "Request %s: Using position-based spans - "
                    "span_starts=%s, cross_span_starts=%s",
                    request_id, span_starts, cross_span_starts,
                )
                stream = app.state.client.chat.completions.create(**chat_params)
            else:
                stream = app.state.client.completions.create(**completion_params)

            # Process stream
            full_text, finish_reason, ttft_seconds = process_stream(
                stream, start_time, request_id, app.state.logger
            )
        except Exception as e:
            logger.error(
                "Request %s: Error streaming to client: %s",
                request_id,
                str(e),
                exc_info=True,
            )
            raise

        # Apply response trimming if enabled
        if config.processing.trim and config.processing.trim.enabled:
            original_text = full_text
            full_text = trim_response(
                full_text,
                config.processing.trim.trim_start,
                config.processing.trim.trim_finish
            )
            if original_text != full_text:
                logger.debug(
                    "Request %s: Response trimmed from %d to %d characters",
                    request_id,
                    len(original_text),
                    len(full_text)
                )

        # Calculate usage
        usage_info = calculate_usage_info(
            full_prompt, full_text, ttft_seconds, app.state.tokenizer
        )

        # Decode full_prompt tokens back to text
        prompt_sent = app.state.tokenizer.decode(full_prompt, skip_special_tokens=False)
        # Log complete request with statistics
        app.state.logger.log_complete(
            request_id,
            request_data={
                "model": model,
                "messages_received": messages,
                "prompt_sent": prompt_sent,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            response_data={"content": full_text, "finish_reason": finish_reason},
            metrics_data=usage_info,
        )

        # Build and return response
        return build_response(request_id, full_text, finish_reason, usage_info)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = app.state.config.server.host
    port = app.state.config.server.port

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)