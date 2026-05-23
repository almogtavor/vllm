# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the render server's /v1/chat/completions/parse handler.

Mocks the configured reasoning + tool parsers and verifies parse_chat_output
orchestrates them: reasoning is split first, then tool calls are extracted
from the post-reasoning content (mirroring /v1/chat/completions).
"""

from unittest.mock import Mock

import pytest

from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.serve.render.serving import (
    OpenAIServingRender,
    ParseRequest,
)


def _handler(reasoning_cls=None, tool_parser_factory=None):
    """Build a handler with only the attributes parse_chat_output touches."""
    h = object.__new__(OpenAIServingRender)
    h.reasoning_parser_cls = reasoning_cls
    h.tool_parser = tool_parser_factory
    h.model_config = Mock(model="test-model")
    h.renderer = Mock()
    h.renderer.get_tokenizer = Mock(return_value=Mock())
    return h


@pytest.mark.asyncio
async def test_splits_reasoning_then_extracts_tools_from_content():
    # reasoning parser returns (reasoning, content)
    rparser = Mock()
    rparser.extract_reasoning.return_value = ("thinking...", "<minimax:tool_call>...")
    reasoning_cls = Mock(return_value=rparser)

    # tool parser extracts a call from the (post-reasoning) content
    tparser = Mock()
    tparser.extract_tool_calls.return_value = ExtractedToolCallInformation(
        tools_called=True,
        tool_calls=[
            ToolCall(function=FunctionCall(name="bash", arguments='{"command":"ls"}'))
        ],
        content=None,
    )
    tool_factory = Mock(return_value=tparser)

    h = _handler(reasoning_cls, tool_factory)
    resp = await h.parse_chat_output(
        ParseRequest(
            text="raw", tools=[{"type": "function", "function": {"name": "bash"}}]
        )
    )

    assert resp.reasoning_content == "thinking..."
    assert resp.tools_called is True
    assert resp.tool_calls[0].function.name == "bash"
    # tool calls are extracted from the reasoning-stripped content, not raw text
    tparser.extract_tool_calls.assert_called_once()
    assert tparser.extract_tool_calls.call_args.args[0] == "<minimax:tool_call>..."


@pytest.mark.asyncio
async def test_plain_text_no_reasoning_no_tools():
    h = _handler(reasoning_cls=None, tool_parser_factory=None)
    resp = await h.parse_chat_output(ParseRequest(text="just an answer"))
    assert resp.reasoning_content is None
    assert resp.content == "just an answer"
    assert resp.tools_called is False
    assert resp.tool_calls == []


@pytest.mark.asyncio
async def test_reasoning_only_no_tool_parser():
    rparser = Mock()
    rparser.extract_reasoning.return_value = ("my reasoning", "the answer")
    h = _handler(reasoning_cls=Mock(return_value=rparser), tool_parser_factory=None)
    resp = await h.parse_chat_output(
        ParseRequest(text="my reasoning</think>the answer")
    )
    assert resp.reasoning_content == "my reasoning"
    assert resp.content == "the answer"
    assert resp.tools_called is False
