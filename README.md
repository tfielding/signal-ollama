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
- Commands: `/model`, `/model <name>`, `/reset`, `/status` (model, memory, what is running now; answers instantly even mid-reply), `/stats` (speed of the last reply), `/help`
- Messages from one person are answered in order; different people don't block each other
- Reconnects to Signal automatically

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
| `SYSTEM_PROMPT` | short plain-text assistant prompt | |
| `OLLAMA_TIMEOUT` | `5m` | |
| `DATA_DIR` | `/data` | Chat history (one JSON file per chat) |

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

## Notes

- signal-cli-rest-api has no authentication. Keep port 8080 off the internet; ideally don't
  publish it at all and let containers talk over the Docker network.
- In json-rpc mode, messages that arrive while signal-ollama is down are not replayed.
- Chat history is stored unencrypted in `DATA_DIR`.

## Develop

```bash
go test -race ./...
```
