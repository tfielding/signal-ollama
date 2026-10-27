// Package signal talks to bbernhard/signal-cli-rest-api running in MODE=json-rpc.
package signal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coder/websocket"
)

// Envelope is the subset of signal-cli's receive payload we care about.
type Envelope struct {
	Source       string          `json:"source"`
	SourceNumber string          `json:"sourceNumber"`
	SourceUUID   string          `json:"sourceUuid"`
	SourceName   string          `json:"sourceName"`
	Timestamp    int64           `json:"timestamp"`
	DataMessage  *DataMessage    `json:"dataMessage"`
	SyncMessage  json.RawMessage `json:"syncMessage"`
}

type DataMessage struct {
	Timestamp int64      `json:"timestamp"`
	Message   string     `json:"message"`
	GroupInfo *GroupInfo `json:"groupInfo"`
}

type GroupInfo struct {
	GroupID string `json:"groupId"`
}

type Client struct {
	base   string
	number string
	http   *http.Client
}

func New(baseURL, number string) *Client {
	return &Client{base: baseURL, number: number, http: &http.Client{Timeout: 30 * time.Second}}
}

// Listen connects to the receive websocket and calls handle for every envelope.
// It reconnects with backoff until ctx is cancelled.
func (c *Client) Listen(ctx context.Context, handle func(Envelope)) error {
	wsURL := strings.Replace(c.base, "http", "ws", 1) + "/v1/receive/" + url.PathEscape(c.number)
	backoff := time.Second
	for {
		err := c.listenOnce(ctx, wsURL, handle, func() { backoff = time.Second })
		if ctx.Err() != nil {
			return ctx.Err()
		}
		slog.Warn("signal websocket disconnected, retrying", "err", err, "in", backoff)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, 30*time.Second)
	}
}

func (c *Client) listenOnce(ctx context.Context, wsURL string, handle func(Envelope), connected func()) error {
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		return err
	}
	defer conn.CloseNow()
	conn.SetReadLimit(4 << 20)
	slog.Info("connected to signal", "url", wsURL)
	connected()

	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var msg struct {
			Envelope Envelope `json:"envelope"`
		}
		if err := json.Unmarshal(data, &msg); err != nil {
			slog.Warn("skipping unparseable message", "err", err)
			continue
		}
		handle(msg.Envelope)
	}
}

// Send sends a message to a single recipient. With styled set, the container
// turns **bold**, *italic*, ~strike~, `mono` and ||spoiler|| into Signal formatting.
func (c *Client) Send(ctx context.Context, recipient, text string, styled bool) error {
	mode := "normal"
	if styled {
		mode = "styled"
	}
	return c.do(ctx, http.MethodPost, "/v2/send", map[string]any{
		"number":     c.number,
		"recipients": []string{recipient},
		"message":    text,
		"text_mode":  mode,
	})
}

// Typing starts or stops the typing indicator shown to recipient.
func (c *Client) Typing(ctx context.Context, recipient string, on bool) error {
	method := http.MethodPut
	if !on {
		method = http.MethodDelete
	}
	return c.do(ctx, method, "/v1/typing-indicator/"+url.PathEscape(c.number), map[string]string{"recipient": recipient})
}

func (c *Client) do(ctx context.Context, method, path string, body any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+path, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("%s %s: %s: %s", method, path, resp.Status, strings.TrimSpace(string(msg)))
	}
	return nil
}
