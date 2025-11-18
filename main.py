"""Entry point for querying AWS costs via Gemini + MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
from copy import deepcopy
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.genai.types import FunctionDeclaration, Tool
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client

load_dotenv()

LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPT = "Give me cost report for last month"
EXIT_COMMANDS = {"exit", "quit", ":q"}


def _clean_schema(schema: Any) -> Any:
    """Return a Gemini-compatible JSON schema, stripping refs and extras."""

    if not isinstance(schema, dict):
        return schema

    working = deepcopy(schema)
    definitions = working.pop("$defs", {})
    working.pop("$schema", None)

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in definitions:
                        return resolve(deepcopy(definitions[def_name]))
                LOGGER.debug("Unresolved $ref '%s' replaced with fallback schema", ref_path)
                return {"type": "object"}

            cleaned: dict[str, Any] = {}
            for key, value in node.items():
                if key in {"additionalProperties", "$defs", "$schema"}:
                    continue
                cleaned[key] = resolve(value)
            return cleaned

        if isinstance(node, list):
            return [resolve(item) for item in node]

        return node

    return resolve(working)


def convert_mcp_tools_to_gemini(mcp_tools: Iterable[mcp_types.Tool]) -> list[Tool]:
    """Translate MCP tool metadata into Gemini Tool declarations."""

    declarations: list[FunctionDeclaration] = []
    for tool in mcp_tools:
        raw_schema = getattr(tool, "inputSchema", None) or {"type": "object"}
        parameters = _clean_schema(raw_schema)
        declarations.append(
            FunctionDeclaration(
                name=tool.name,
                description=tool.description or f"MCP tool {tool.name}",
                parameters=parameters,
            )
        )

    if not declarations:
        return []

    return [Tool(function_declarations=declarations)]


def _build_server_params() -> StdioServerParameters:
    """Create MCP server parameters, respecting environment overrides."""

    command = os.getenv("MCP_SERVER_COMMAND", "docker")
    env_file = os.getenv("MCP_ENV_FILE", "/home/kaushik/.aws/mcp.env")
    docker_image = os.getenv("MCP_DOCKER_IMAGE", "cost-explorer-mcp")

    args_from_env = os.getenv("MCP_SERVER_ARGS")
    if args_from_env:
        args = shlex.split(args_from_env)
    else:
        args = ["run", "--rm", "-i", "--env-file", env_file, docker_image]

    return StdioServerParameters(command=command, args=args)


def _build_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; add it to your environment or .env file")
    return genai.Client(api_key=api_key)


def _generate(
    client: genai.Client,
    contents: Sequence[genai_types.Content],
    tools: Sequence[Tool],
    model: str,
) -> genai_types.GenerateContentResponse:
    return client.models.generate_content(
        model=model,
        contents=list(contents),
        config=genai_types.GenerateContentConfig(tools=list(tools)),
    )


def _candidate_function_calls(candidate: genai_types.Candidate) -> list[genai_types.FunctionCall]:
    calls: list[genai_types.FunctionCall] = []
    if candidate.content is None:
        return calls
    for part in candidate.content.parts:
        if part.function_call:
            calls.append(part.function_call)
    return calls


def _render_candidate_text(candidate: genai_types.Candidate) -> str:
    if candidate.content is None:
        return ""
    texts = [part.text.strip() for part in candidate.content.parts if part.text]
    return "\n".join(filter(None, texts)).strip()


def _content_block_to_text(block: mcp_types.ContentBlock) -> str:
    if isinstance(block, mcp_types.TextContent):
        return block.text
    try:
        return json.dumps(block.model_dump(exclude_none=True), default=str)
    except AttributeError:
        return str(block)


def _tool_result_payload(result: mcp_types.CallToolResult) -> dict[str, Any]:
    text_chunks = [_content_block_to_text(block) for block in result.content or []]
    payload: dict[str, Any] = {}
    if text_chunks:
        payload["text"] = "\n".join(chunk for chunk in text_chunks if chunk).strip()
    if result.structuredContent:
        payload["structured"] = result.structuredContent
    if not payload:
        payload["text"] = ""
    return payload


async def _invoke_tool(
    session: ClientSession,
    function_call: genai_types.FunctionCall,
) -> genai_types.Part:
    LOGGER.info("Calling MCP tool '%s'", function_call.name)
    arguments = dict(function_call.args or {})
    result = await session.call_tool(function_call.name, arguments)
    return genai_types.Part.from_function_response(
        name=function_call.name,
        response=_tool_result_payload(result),
    )


async def _run_chat_turn(
    session: ClientSession,
    client: genai.Client,
    conversation: list[genai_types.Content],
    gemini_tools: Sequence[Tool],
    model: str,
) -> str:
    response = _generate(client, conversation, gemini_tools, model)

    while True:
        if not response.candidates:
            raise RuntimeError("Gemini returned no candidates")

        candidate = response.candidates[0]
        function_calls = _candidate_function_calls(candidate)

        if not function_calls:
            assistant_text = _render_candidate_text(candidate)
            parts = candidate.content.parts if candidate.content and candidate.content.parts else []
            if parts:
                conversation.append(genai_types.Content(role="model", parts=parts))
            else:
                conversation.append(
                    genai_types.Content(
                        role="model", parts=[genai_types.Part.from_text(text=assistant_text)]
                    )
                )
            return assistant_text

        if candidate.content and candidate.content.parts:
            conversation.append(genai_types.Content(role="model", parts=candidate.content.parts))

        for call in function_calls:
            tool_part = await _invoke_tool(session, call)
            conversation.append(genai_types.Content(role="user", parts=[tool_part]))

        response = _generate(client, conversation, gemini_tools, model)


async def query_aws_cost(prompt: str) -> str:
    """Query Gemini for AWS cost insights, fulfilling tool calls via MCP."""

    server_params = _build_server_params()
    client = _build_gemini_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            gemini_tools = convert_mcp_tools_to_gemini(tools_response.tools)

            if not gemini_tools:
                raise RuntimeError("No MCP tools available to expose to Gemini")

            conversation: list[genai_types.Content] = [
                genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=prompt)])
            ]

            return await _run_chat_turn(session, client, conversation, gemini_tools, model)


async def _read_user_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def interactive_chat(initial_prompt: str | None = None) -> None:
    """Start an interactive multi-turn chat with Gemini and MCP."""

    server_params = _build_server_params()
    client = _build_gemini_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            gemini_tools = convert_mcp_tools_to_gemini(tools_response.tools)

            if not gemini_tools:
                raise RuntimeError("No MCP tools available to expose to Gemini")

            conversation: list[genai_types.Content] = []
            print("Starting interactive AWS cost chat. Type 'exit' to leave.")

            if initial_prompt:
                print(f"You: {initial_prompt}")
                conversation.append(
                    genai_types.Content(
                        role="user", parts=[genai_types.Part.from_text(text=initial_prompt)]
                    )
                )
                try:
                    assistant_text = await _run_chat_turn(
                        session, client, conversation, gemini_tools, model
                    )
                except Exception as exc:  # pragma: no cover - interactive guardrail
                    LOGGER.exception("Error handling initial prompt")
                    print(f"[ERROR] {exc}")
                else:
                    if assistant_text:
                        print(f"Assistant: {assistant_text}")

            while True:
                try:
                    user_input = (await _read_user_input("You: ")).strip()
                except EOFError:
                    print()
                    break

                if not user_input:
                    continue

                if user_input.lower() in EXIT_COMMANDS:
                    break

                conversation.append(
                    genai_types.Content(
                        role="user", parts=[genai_types.Part.from_text(text=user_input)]
                    )
                )

                try:
                    assistant_text = await _run_chat_turn(
                        session, client, conversation, gemini_tools, model
                    )
                except Exception as exc:  # pragma: no cover - interactive guardrail
                    LOGGER.exception("Error during chat turn")
                    print(f"[ERROR] {exc}")
                    continue

                if assistant_text:
                    print(f"Assistant: {assistant_text}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Gemini about your AWS bill via MCP")
    parser.add_argument(
        "--prompt",
        "-p",
        default=None,
        help="Question to ask Gemini (defaults to last month's cost report)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start an interactive chat session after the first response",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompt = args.prompt or DEFAULT_PROMPT
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    if args.interactive:
        try:
            asyncio.run(interactive_chat(initial_prompt=args.prompt))
        except KeyboardInterrupt:
            LOGGER.info("Cancelled by user")
        return

    try:
        result = asyncio.run(query_aws_cost(prompt))
    except KeyboardInterrupt:
        LOGGER.info("Cancelled by user")
        return

    if result:
        print(result)


if __name__ == "__main__":
    main()