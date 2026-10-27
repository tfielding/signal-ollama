// Package ollama is a minimal client for the Ollama HTTP API.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Message struct {
	Role      string     `json:"role"` // system, user, assistant, tool
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"`
	Images    [][]byte   `json:"images,omitempty"` // sent as base64
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"` // on role "tool" messages
}

// Tool is a function the model may call.
type Tool struct {
	Type     string       `json:"type"` // "function"
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters"`
}

type ToolCall struct {
	Function struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"function"`
}

type Client struct {
	base string
	http *http.Client
}

func New(baseURL string, timeout time.Duration) *Client {
	return &Client{base: baseURL, http: &http.Client{Timeout: timeout}}
}

// Stats are the timings Ollama reports with each reply (durations are nanoseconds in JSON).
type Stats struct {
	TotalDuration      time.Duration `json:"total_duration"`
	LoadDuration       time.Duration `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
}

func (s Stats) TokensPerSecond() float64       { return rate(s.EvalCount, s.EvalDuration) }
func (s Stats) PromptTokensPerSecond() float64 { return rate(s.PromptEvalCount, s.PromptEvalDuration) }

// Add accumulates another call's stats (for multi-step tool use).
func (s *Stats) Add(o Stats) {
	s.TotalDuration += o.TotalDuration
	s.LoadDuration += o.LoadDuration
	s.PromptEvalCount += o.PromptEvalCount
	s.PromptEvalDuration += o.PromptEvalDuration
	s.EvalCount += o.EvalCount
	s.EvalDuration += o.EvalDuration
}

func rate(n int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}

type Result struct {
	Content   string
	Thinking  string
	ToolCalls []ToolCall
	Stats     Stats
}

// Chat sends the conversation and returns the assistant's reply.
// think is nil (model default), a bool, or a level string such as "low".
// tools may be nil; if the reply asks for tool calls they are in Result.ToolCalls.
func (c *Client) Chat(ctx context.Context, model string, msgs []Message, think any, tools []Tool) (Result, error) {
	req := map[string]any{"model": model, "messages": msgs, "stream": false}
	if think != nil {
		req["think"] = think
	}
	if len(tools) > 0 {
		req["tools"] = tools
	}
	body, _ := json.Marshal(req)
	var out struct {
		Message Message `json:"message"`
		Stats
	}
	if err := c.do(ctx, http.MethodPost, "/api/chat", body, &out); err != nil {
		return Result{}, err
	}
	return Result{
		Content:   strings.TrimSpace(out.Message.Content),
		Thinking:  out.Message.Thinking,
		ToolCalls: out.Message.ToolCalls,
		Stats:     out.Stats,
	}, nil
}

// Ping checks that Ollama is reachable, failing fast (it doesn't wait for a model).
func (c *Client) Ping(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	var out struct {
		Version string `json:"version"`
	}
	return c.do(ctx, http.MethodGet, "/api/version", nil, &out)
}

// ThinkInfo describes a model's thinking options, from /api/show.
// Values use "on"/"off" for true/false and keep level names (e.g. "low") as-is.
type ThinkInfo struct {
	Known   bool // false if Ollama didn't report thinking metadata for this model
	Values  []string
	Default string
}

func (t ThinkInfo) Supports(v string) bool {
	for _, x := range t.Values {
		if x == v {
			return true
		}
	}
	return false
}

// ModelInfo is what /api/show tells us about a model.
type ModelInfo struct {
	Think        ThinkInfo
	Capabilities []string // e.g. "completion", "tools", "vision", "thinking"
}

func (m ModelInfo) Has(capability string) bool {
	for _, c := range m.Capabilities {
		if c == capability {
			return true
		}
	}
	return false
}

// Info returns thinking options and capabilities for model.
func (c *Client) Info(ctx context.Context, model string) (ModelInfo, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	body, _ := json.Marshal(map[string]string{"model": model})
	var out struct {
		Capabilities []string `json:"capabilities"`
		Thinking     *struct {
			Values  []any `json:"values"`
			Default any   `json:"default"`
		} `json:"thinking"`
	}
	if err := c.do(ctx, http.MethodPost, "/api/show", body, &out); err != nil {
		return ModelInfo{}, err
	}
	info := ModelInfo{Capabilities: out.Capabilities}
	if out.Thinking == nil {
		return info, nil
	}
	info.Think.Default = thinkName(out.Thinking.Default)
	for _, v := range out.Thinking.Values {
		if n := thinkName(v); n != "" {
			info.Think.Values = append(info.Think.Values, n)
		}
	}
	info.Think.Known = len(info.Think.Values) > 0
	return info, nil
}

func thinkName(v any) string {
	switch x := v.(type) {
	case bool:
		if x {
			return "on"
		}
		return "off"
	case string:
		return x
	}
	return ""
}

// ThinkValue converts a setting ("on", "off", "low", ...) to the API value; "" means model default.
func ThinkValue(setting string) any {
	switch setting {
	case "":
		return nil
	case "on":
		return true
	case "off":
		return false
	}
	return setting
}

// Models lists the models installed on the Ollama server.
func (c *Client) Models(ctx context.Context) ([]string, error) {
	var out struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := c.do(ctx, http.MethodGet, "/api/tags", nil, &out); err != nil {
		return nil, err
	}
	names := make([]string, len(out.Models))
	for i, m := range out.Models {
		names[i] = m.Name
	}
	return names, nil
}

func (c *Client) do(ctx context.Context, method, path string, body []byte, out any) error {
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("ollama %s: %s: %s", path, resp.Status, strings.TrimSpace(string(msg)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}
