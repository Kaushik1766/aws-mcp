## AWS Cost via Gemini + MCP

This tiny utility bridges your local Model Context Protocol (MCP) cost-explorer server with Gemini so you can ask natural-language questions (for example, "How much did I spend last month?") and let Gemini call the MCP tool to fetch authoritative AWS billing data.

### Prerequisites

- Python 3.13 (a virtual environment is recommended)
- Docker installed locally (the default setup shells out to `docker run ... cost-explorer-mcp`)
- Access to the [Gemini API](https://ai.google.dev) and a valid API key
- An MCP cost explorer server image plus an `.env` file containing your AWS credentials/secrets (referenced by `MCP_ENV_FILE`)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Environment variables

Create a `.env` file (or export vars in your shell) with at least:

- `GEMINI_API_KEY` – required; your Gemini key
- `MCP_ENV_FILE` – optional; path to the env file that should be passed through to the MCP Docker container (defaults to `/home/kaushik/.aws/mcp.env`)
- `MCP_DOCKER_IMAGE` – optional; defaults to `cost-explorer-mcp`
- `MCP_SERVER_COMMAND` / `MCP_SERVER_ARGS` – optional; override the entire MCP launch command/args if you don't want the default `docker run --rm -i --env-file ... cost-explorer-mcp`
- `GEMINI_MODEL` – optional; change the Gemini model (defaults to `gemini-2.5-pro`)

### Usage

```bash
python main.py --prompt "Give me an AWS cost breakdown for the last quarter"
```

The script will:

1. Start/attach to the MCP server specified by your env vars.
2. Fetch its tool schema and translate it into Gemini function declarations.
3. Ask Gemini your prompt; when the model emits a tool call, the script executes it through MCP and feeds the results back to Gemini.
4. Print the final natural-language response.

Use `--log-level DEBUG` for more insight into the request/response flow.
