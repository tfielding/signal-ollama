package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"signal-ollama/internal/bot"
	"signal-ollama/internal/ollama"
	sig "signal-ollama/internal/signal"
	"signal-ollama/internal/store"
	"signal-ollama/internal/toolbox"
)

var testJPEG = []byte{0xFF, 0xD8, 0xFF, 0xE0, 'j', 'p', 'g', 0xFF, 0xD9}

// mcpServer is a stand-in camera MCP server: two read-only tools and one that isn't.
func mcpServer(t *testing.T, token string) *httptest.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: "cams", Version: "1"}, nil)
	ro := &mcp.ToolAnnotations{ReadOnlyHint: true}
	type snapIn struct {
		Camera string `json:"camera"`
	}
	mcp.AddTool(s, &mcp.Tool{Name: "get_snapshot", Description: "Take a snapshot", Annotations: ro},
		func(_ context.Context, _ *mcp.CallToolRequest, in snapIn) (*mcp.CallToolResult, any, error) {
			if in.Camera != "driveway" {
				return nil, nil, &toolErr{"unknown camera " + in.Camera}
			}
			return &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: "Snapshot from driveway"},
				&mcp.ImageContent{Data: testJPEG, MIMEType: "image/jpeg"},
			}}, nil, nil
		})
	mcp.AddTool(s, &mcp.Tool{Name: "list_cameras", Annotations: ro},
		func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, map[string][]string, error) {
			return nil, map[string][]string{"cameras": {"driveway"}}, nil
		})
	mcp.AddTool(s, &mcp.Tool{Name: "delete_stream"}, // not read-only: must be hidden
		func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, any, error) {
			t.Error("delete_stream must never be called")
			return nil, nil, nil
		})
	h := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return s }, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+token {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		h.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

type toolErr struct{ s string }

func (e *toolErr) Error() string { return e.s }

type chatReq struct {
	Model    string           `json:"model"`
	Messages []ollama.Message `json:"messages"`
	Tools    []ollama.Tool    `json:"tools"`
}

// fakeOllama asks for a snapshot on the first turn, then answers using the tool result.
func fakeOllama(t *testing.T, capabilities []string, reqs *[]chatReq, mu *sync.Mutex) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/version":
			w.Write([]byte(`{"version":"t"}`))
		case "/api/show":
			json.NewEncoder(w).Encode(map[string]any{"capabilities": capabilities})
		case "/api/chat":
			var req chatReq
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			*reqs = append(*reqs, req)
			mu.Unlock()
			last := req.Messages[len(req.Messages)-1]
			if last.Role == "user" {
				json.NewEncoder(w).Encode(map[string]any{
					"message":    map[string]any{"role": "assistant", "content": "", "thinking": "need a snapshot", "tool_calls": []map[string]any{{"function": map[string]any{"name": "get_snapshot", "arguments": map[string]any{"camera": "driveway"}}}}},
					"eval_count": 10, "eval_duration": 1e9,
				})
				return
			}
			json.NewEncoder(w).Encode(map[string]any{
				"message":    map[string]any{"role": "assistant", "content": "Here's the **driveway**."},
				"eval_count": 20, "eval_duration": 1e9,
			})
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func fakeSignal(t *testing.T, botNum, me string, sends chan map[string]any) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/v1/receive/"):
			c, err := websocket.Accept(w, r, nil)
			if err != nil {
				return
			}
			defer c.CloseNow()
			c.Write(r.Context(), websocket.MessageText, []byte(`{"envelope":{"source":"`+me+`","sourceNumber":"`+me+`","dataMessage":{"message":"show me the driveway"}},"account":"`+botNum+`"}`))
			<-r.Context().Done()
		case r.URL.Path == "/v2/send":
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			w.WriteHeader(http.StatusCreated)
			sends <- body
		default:
			w.WriteHeader(http.StatusNoContent) // typing indicator
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func runToolScenario(t *testing.T, capabilities []string) (map[string]any, []chatReq, *store.Store) {
	t.Helper()
	const botNum, me = "+15550000009", "+15550000001"
	var mu sync.Mutex
	var reqs []chatReq
	sends := make(chan map[string]any, 4)

	cams := mcpServer(t, "tok")
	llmSrv := fakeOllama(t, capabilities, &reqs, &mu)
	sigSrv := fakeSignal(t, botNum, me, sends)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	box := toolbox.New([]toolbox.Server{{Name: "cams", URL: cams.URL, Token: "tok"}}, toolbox.Options{})
	defer box.Close()
	st, _ := store.New(t.TempDir())
	sc := sig.New(sigSrv.URL, botNum)
	b := bot.New(ctx, sc, ollama.New(llmSrv.URL, 5*time.Second), st, bot.Options{
		Tools: box, OwnNumber: botNum, Allowed: map[string]bool{me: true}, DefaultModel: "m", HistoryLimit: 10,
	})
	done := make(chan struct{})
	go func() { sc.Listen(ctx, b.Handle); close(done) }()

	var sent map[string]any
	select {
	case sent = <-sends:
	case <-time.After(5 * time.Second):
		t.Fatal("no reply sent")
	}
	cancel()
	<-done
	b.Wait()
	mu.Lock()
	defer mu.Unlock()
	return sent, append([]chatReq(nil), reqs...), st
}

func TestToolSnapshotSentAsAttachment(t *testing.T) {
	sent, reqs, st := runToolScenario(t, []string{"completion", "tools"})

	// Only read-only tools are offered to the model.
	var offered []string
	for _, tool := range reqs[0].Tools {
		offered = append(offered, tool.Function.Name)
	}
	if strings.Join(offered, ",") != "get_snapshot,list_cameras" {
		t.Fatalf("offered tools: %v", offered)
	}

	// Tool guidance is added after the main system prompt when tools are offered.
	if len(reqs[0].Messages) < 2 || reqs[0].Messages[0].Role != "system" || !strings.Contains(reqs[0].Messages[0].Content, "call the tools again") {
		t.Fatalf("tool guidance missing: %+v", reqs[0].Messages)
	}

	// Second call carries the assistant tool call and the tool result (text only: no vision).
	if len(reqs) != 2 {
		t.Fatalf("expected 2 model calls, got %d", len(reqs))
	}
	msgs := reqs[1].Messages
	call, result := msgs[len(msgs)-2], msgs[len(msgs)-1]
	if call.Role != "assistant" || len(call.ToolCalls) != 1 || call.Thinking != "need a snapshot" {
		t.Fatalf("assistant tool-call message: %+v", call)
	}
	if result.Role != "tool" || result.ToolName != "get_snapshot" || len(result.Images) != 0 ||
		!strings.Contains(result.Content, "Snapshot from driveway") || !strings.Contains(result.Content, "you cannot see them") {
		t.Fatalf("tool message: %+v", result)
	}

	// The reply is styled and carries the JPEG.
	att, _ := sent["base64_attachments"].([]any)
	want := "data:image/jpeg;filename=image-1.jpg;base64," + base64.StdEncoding.EncodeToString(testJPEG)
	if len(att) != 1 || att[0] != want {
		t.Fatalf("attachments: %v", sent["base64_attachments"])
	}
	if sent["message"] != "Here's the **driveway**." || sent["text_mode"] != "styled" {
		t.Fatalf("message: %v", sent)
	}

	c, _ := st.Load("+15550000001")
	if len(c.History) != 2 || !strings.Contains(c.History[1].Content, "1 image(s) were sent") {
		t.Fatalf("history: %+v", c.History)
	}
	if c.Last.Stats.EvalCount != 30 {
		t.Fatalf("stats not summed across calls: %+v", c.Last.Stats)
	}
}

func TestToolImageGoesToVisionModel(t *testing.T) {
	_, reqs, _ := runToolScenario(t, []string{"completion", "tools", "vision"})
	result := reqs[1].Messages[len(reqs[1].Messages)-1]
	if len(result.Images) != 1 || string(result.Images[0]) != string(testJPEG) {
		t.Fatalf("vision model should get the image: %+v", result)
	}
}

func TestNoToolsForModelWithoutToolSupport(t *testing.T) {
	sent, reqs, _ := runToolScenario(t, []string{"completion"})
	if len(reqs[0].Tools) != 0 {
		t.Fatalf("tools offered to a model without tool support: %v", reqs[0].Tools)
	}
	for _, m := range reqs[0].Messages {
		if strings.Contains(m.Content, "call the tools again") {
			t.Fatal("tool guidance added without tools")
		}
	}
	if sent["base64_attachments"] != nil {
		t.Fatalf("unexpected attachments")
	}
}

// TestRepeatedToolFailuresStopEarly: a model that keeps calling a tool wrongly is cut off
// after maxToolFailures, told to answer, and a blank answer is explained to the user.
func TestRepeatedToolFailuresStopEarly(t *testing.T) {
	const botNum, me = "+15550000009", "+15550000001"
	var mu sync.Mutex
	var reqs []chatReq
	llm := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/version":
			w.Write([]byte(`{"version":"t"}`))
		case "/api/show":
			w.Write([]byte(`{"capabilities":["completion","tools"]}`))
		case "/api/chat":
			var req chatReq
			json.NewDecoder(r.Body).Decode(&req)
			mu.Lock()
			reqs = append(reqs, req)
			mu.Unlock()
			if len(req.Tools) == 0 { // forced final answer: come back blank, like gpt-oss did
				w.Write([]byte(`{"message":{"role":"assistant","content":""}}`))
				return
			}
			w.Write([]byte(`{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_snapshot","arguments":{"camera":"attic"}}}]}}`))
		}
	}))
	defer llm.Close()
	sends := make(chan map[string]any, 4)
	sigSrv := fakeSignal(t, botNum, me, sends)
	cams := mcpServer(t, "tok")

	ctx, cancel := context.WithCancel(context.Background())
	box := toolbox.New([]toolbox.Server{{Name: "cams", URL: cams.URL, Token: "tok"}}, toolbox.Options{})
	defer box.Close()
	st, _ := store.New(t.TempDir())
	sc := sig.New(sigSrv.URL, botNum)
	b := bot.New(ctx, sc, ollama.New(llm.URL, 5*time.Second), st, bot.Options{
		Tools: box, OwnNumber: botNum, Allowed: map[string]bool{me: true}, DefaultModel: "m", HistoryLimit: 10,
	})
	done := make(chan struct{})
	go func() { sc.Listen(ctx, b.Handle); close(done) }()
	var sent map[string]any
	select {
	case sent = <-sends:
	case <-time.After(5 * time.Second):
		t.Fatal("no reply")
	}
	cancel()
	<-done
	b.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(reqs) != 4 { // 3 failing tool rounds, then one forced answer
		t.Fatalf("model calls: %d", len(reqs))
	}
	last := reqs[3]
	if len(last.Tools) != 0 || !strings.Contains(last.Messages[len(last.Messages)-1].Content, "do not guess") {
		t.Fatalf("final request: tools=%d last=%+v", len(last.Tools), last.Messages[len(last.Messages)-1])
	}
	if msg, _ := sent["message"].(string); !strings.Contains(msg, "3 tool call(s) failed") {
		t.Fatalf("reply: %q", msg)
	}
}
