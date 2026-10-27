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
	Role    string `json:"role"` // system, user, assistant
	Content string `json:"content"`
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

func rate(n int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}

type Result struct {
	Content string
	Stats   Stats
}

// Chat sends the conversation and returns the assistant's reply.
func (c *Client) Chat(ctx context.Context, model string, msgs []Message) (Result, error) {
	body, _ := json.Marshal(map[string]any{"model": model, "messages": msgs, "stream": false})
	var out struct {
		Message Message `json:"message"`
		Stats
	}
	if err := c.do(ctx, http.MethodPost, "/api/chat", body, &out); err != nil {
		return Result{}, err
	}
	return Result{Content: strings.TrimSpace(out.Message.Content), Stats: out.Stats}, nil
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
