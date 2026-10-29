# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-29

### Added
- Chat with Ollama models over Signal via signal-cli-rest-api (json-rpc mode).
- Allowlist of senders; group chats and the account's own messages are ignored.
- Per-chat memory saved to disk, typing indicator, Markdown converted to Signal formatting.
- Commands: `/model`, `/reset`, `/status`, `/think`, `/stop`, `/stats`, `/help`.
- Tools from MCP servers (Streamable HTTP): read-only by default, optional allowlist, per-server
  tokens and hints, server instructions passed to the model, images sent as attachments.
- Safeguards for tool use: invented/blank arguments removed, read-only calls retried without
  failing optional parameters, failure and round limits, explained empty answers.
- Fast "Ollama is offline" reply; `-version` flag.
