# signal-ollama

Chat with your own Ollama models over Signal. A small Go service that sits next to
[bbernhard/signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) and
answers direct messages from an allowlist of phone numbers.

```
Signal app ⇄ signal-cli-rest-api (json-rpc) ⇄ signal-ollama ⇄ Ollama
                 ws /v1/receive                          /api/chat
                 POST /v2/send
```

## What it answers (and what it ignores)

| Message | Behavior |
|---|---|
| Direct message from a number in `ALLOWED_SENDERS` | Answered |
| Messages in group chats | Ignored |
| Direct message from anyone else | Ignored (logged) |
| Messages the account sends itself (alerts, sync messages) | Ignored |

It only reads from the receive websocket and sends direct replies, so other services that
use the same signal-cli-rest-api account to *send* (alerts etc.) are unaffected.

## Features

- Per-chat conversation memory, saved to disk (`DATA_DIR`), last `HISTORY_LIMIT` messages
- Typing indicator while the model is thinking
- Markdown from the model is converted to Signal formatting (bold, italic, monospace); headings become bold lines, tables become label/value lines
- Commands: `/model`, `/model <name>`, `/reset`, `/status` (model, thinking, memory, whether Ollama is online, what is running now; answers instantly even mid-reply), `/think` (reasoning on/off or low/medium/high, per chat), `/stop` (cancel the reply in progress and anything waiting), `/stats` (speed of the last reply), `/help`
- Messages from one person are answered in order; different people don't block each other
- Reconnects to Signal automatically
- Optional tools from MCP servers (e.g. camera snapshots); images that tools return are sent to you as attachments
- If Ollama is unreachable (e.g. the machine is asleep), replies right away saying so instead of waiting for a timeout

## Requirements

- signal-cli-rest-api with `MODE=json-rpc` and a registered number
- Ollama reachable from the container. If Ollama runs on another machine, make it listen
  on the network: `OLLAMA_HOST=0.0.0.0` (macOS app: `launchctl setenv OLLAMA_HOST 0.0.0.0`,
  then restart Ollama).

## Configuration

| Variable | Default | |
|---|---|---|
| `SIGNAL_NUMBER` | required | The number registered in signal-cli-rest-api |
| `ALLOWED_SENDERS` | required | Comma-separated phone numbers (or Signal UUIDs) |
| `SIGNAL_URL` | `http://signal-cli-rest-api:8080` | |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | first installed model | Default model; each chat can switch with `/model` |
| `HISTORY_LIMIT` | `40` | Messages kept per chat |
| `SYSTEM_PROMPT` | short, phone-friendly assistant prompt | Overrides the built-in prompt (including its formatting rules) |
| `OLLAMA_TIMEOUT` | `5m` | |
| `DATA_DIR` | `/data` | Chat history (one JSON file per chat) |
| `MCP_SERVERS` | none | MCP servers to get tools from: `name=url,...`, e.g. `cams=http://go2rtc-mcp:8090/mcp` |
| `MCP_TOKEN_<NAME>` | none | Bearer token for server `<name>` (uppercased, `-` and `.` become `_`), e.g. `MCP_TOKEN_CAMS` |
| `MCP_TOOLS` | all | Optional allowlist of tool names, e.g. `list_cameras,get_snapshot` |
| `MCP_ALLOW_WRITE` | `false` | Also expose tools that are not marked read-only |
| `MCP_MAX_RESULT_CHARS` | `32000` | Max characters of tool output passed to the model (about 8k tokens). Make sure Ollama's context window can hold it |

## Run

With Docker Compose, add the service from `docker-compose.example.yml` next to your
existing `signal-cli-rest-api` service (use `image: ghcr.io/tfielding/signal-ollama:latest`
or `build:`). The image runs as a non-root user (uid 65532), so create the data folder first:

```bash
sudo mkdir -p /opt/signal-ollama && sudo chown 65532:65532 /opt/signal-ollama
docker compose up -d signal-ollama
```

Use `SIGNAL_URL=http://signal-cli-rest-api:8080` (the service name), not `localhost`:
inside a container, `localhost` is the container itself.

The typing indicator only shows if "Typing indicators" is on in your Signal app
(Settings → Privacy).

Locally:

```bash
SIGNAL_URL=http://<signal-host>:8080 SIGNAL_NUMBER=+1... ALLOWED_SENDERS=+1... \
OLLAMA_URL=http://localhost:11434 DATA_DIR=./data go run .
```

## Tools (MCP)

signal-ollama can use tools from any [MCP](https://modelcontextprotocol.io) server that speaks
Streamable HTTP. It has no tool-specific code: point `MCP_SERVERS` at a server and its tools are
offered to the model.

- Only tools marked **read-only** by the server are offered, unless `MCP_ALLOW_WRITE=true`.
  Anything that can change things (locks, alarms, settings) stays hidden by default.
- Tools are only offered to models that support tool calling (Ollama reports this per model).
- Images returned by a tool are sent to you as Signal attachments with the reply. If the current
  model supports vision, it sees them too; otherwise it's told an image was sent.
- The model can make up to 6 rounds of tool calls per message. `/stop` cancels tool calls too.
- `/status` lists the available tools and any server that's unreachable. Tool lists are refreshed
  every minute, and sessions reconnect automatically if a server restarts.

## Notes

- signal-cli-rest-api has no authentication. Keep port 8080 off the internet; ideally don't
  publish it at all and let containers talk over the Docker network.
- In json-rpc mode, messages that arrive while signal-ollama is down are not replayed.
- Chat history is stored unencrypted in `DATA_DIR`.

## Develop

```bash
go test -race ./...
```
