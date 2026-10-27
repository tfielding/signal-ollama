// Package config loads settings from environment variables.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"signal-ollama/internal/toolbox"
)

const defaultSystemPrompt = "You are a helpful assistant chatting over Signal on a phone. " +
	"Keep answers short and conversational unless asked for detail. " +
	"Formatting: you may use **bold**, *italic*, `code` and simple bullet lists. " +
	"Do not use headings, tables or horizontal rules."

type Config struct {
	SignalURL     string          // base URL of signal-cli-rest-api, e.g. http://signal-cli-rest-api:8080
	SignalNumber  string          // the account registered in the container
	OllamaURL     string          // e.g. http://192.168.1.50:11434
	DefaultModel  string          // empty = first model Ollama reports
	Allowed       map[string]bool // phone numbers (or Signal UUIDs) allowed to talk to the bot
	DataDir       string          // where chat history is stored
	HistoryLimit  int             // max messages (user+assistant) kept per chat
	SystemPrompt  string
	OllamaTimeout time.Duration

	MCPServers    []toolbox.Server // from MCP_SERVERS=name=url,...; tokens from MCP_TOKEN_<NAME>
	MCPTools      map[string]bool  // optional allowlist of tool names
	MCPAllowWrite bool             // expose tools not marked read-only
	MCPMaxChars   int              // cap on tool output given to the model
}

func Load() (Config, error) {
	c := Config{
		SignalURL:    strings.TrimRight(env("SIGNAL_URL", "http://signal-cli-rest-api:8080"), "/"),
		SignalNumber: os.Getenv("SIGNAL_NUMBER"),
		OllamaURL:    strings.TrimRight(env("OLLAMA_URL", "http://localhost:11434"), "/"),
		DefaultModel: os.Getenv("OLLAMA_MODEL"),
		Allowed:      map[string]bool{},
		DataDir:      env("DATA_DIR", "/data"),
		SystemPrompt: env("SYSTEM_PROMPT", defaultSystemPrompt),
	}

	if c.SignalNumber == "" {
		return c, fmt.Errorf("SIGNAL_NUMBER is required")
	}
	for _, s := range strings.Split(os.Getenv("ALLOWED_SENDERS"), ",") {
		if s = strings.TrimSpace(s); s != "" {
			c.Allowed[s] = true
		}
	}
	if len(c.Allowed) == 0 {
		return c, fmt.Errorf("ALLOWED_SENDERS is required (comma-separated phone numbers)")
	}

	var err error
	if c.HistoryLimit, err = strconv.Atoi(env("HISTORY_LIMIT", "40")); err != nil || c.HistoryLimit < 2 {
		return c, fmt.Errorf("HISTORY_LIMIT must be an integer >= 2")
	}
	if c.OllamaTimeout, err = time.ParseDuration(env("OLLAMA_TIMEOUT", "5m")); err != nil {
		return c, fmt.Errorf("OLLAMA_TIMEOUT: %w", err)
	}

	if c.MCPServers, err = toolbox.ParseServers(os.Getenv("MCP_SERVERS"), func(name string) string {
		return os.Getenv("MCP_TOKEN_" + strings.ToUpper(strings.NewReplacer("-", "_", ".", "_").Replace(name)))
	}); err != nil {
		return c, err
	}
	c.MCPTools = map[string]bool{}
	for _, t := range strings.Split(os.Getenv("MCP_TOOLS"), ",") {
		if t = strings.TrimSpace(t); t != "" {
			c.MCPTools[t] = true
		}
	}
	if c.MCPMaxChars, err = strconv.Atoi(env("MCP_MAX_RESULT_CHARS", strconv.Itoa(toolbox.DefaultMaxResultChars))); err != nil || c.MCPMaxChars < 1000 {
		return c, fmt.Errorf("MCP_MAX_RESULT_CHARS must be an integer >= 1000")
	}
	if v := os.Getenv("MCP_ALLOW_WRITE"); v != "" {
		if c.MCPAllowWrite, err = strconv.ParseBool(v); err != nil {
			return c, fmt.Errorf("MCP_ALLOW_WRITE must be true or false")
		}
	}
	return c, nil
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
