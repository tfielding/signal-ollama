package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"

	"signal-ollama/internal/bot"
	"signal-ollama/internal/ollama"
	sig "signal-ollama/internal/signal"
	"signal-ollama/internal/store"
)

// TestEndToEnd runs the real clients against fake signal-cli-rest-api and Ollama servers.
func TestEndToEnd(t *testing.T) {
	const botNum, me = "+15550000009", "+15550000001"

	var mu sync.Mutex
	var sends []map[string]any
	var typing []string
	sent := make(chan struct{}, 4)

	// Fake signal-cli-rest-api (json-rpc mode).
	signalSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/receive/"+botNum:
			c, err := websocket.Accept(w, r, nil)
			if err != nil {
				return
			}
			defer c.CloseNow()
			for _, raw := range []string{
				// group chat message: ignored
				`{"envelope":{"source":"` + me + `","sourceNumber":"` + me + `","dataMessage":{"message":"group hi","groupInfo":{"groupId":"g1","type":"DELIVER"}}},"account":"` + botNum + `"}`,
				// our own alert echoed back as a sync message: ignored
				`{"envelope":{"source":"` + botNum + `","sourceNumber":"` + botNum + `","syncMessage":{"sentMessage":{"message":"ALERT"}}},"account":"` + botNum + `"}`,
				// direct message from me: answered
				`{"envelope":{"source":"` + me + `","sourceNumber":"` + me + `","sourceUuid":"u1","dataMessage":{"timestamp":1,"message":"what is 2+2"}},"account":"` + botNum + `"}`,
			} {
				c.Write(r.Context(), websocket.MessageText, []byte(raw))
			}
			<-r.Context().Done()
		case r.URL.Path == "/v2/send" && r.Method == http.MethodPost:
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			mu.Lock()
			sends = append(sends, body)
			mu.Unlock()
			w.WriteHeader(http.StatusCreated)
			sent <- struct{}{}
		case r.URL.Path == "/v1/typing-indicator/"+botNum:
			mu.Lock()
			typing = append(typing, r.Method)
			mu.Unlock()
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Errorf("unexpected signal request %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer signalSrv.Close()

	// Fake Ollama.
	ollamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/chat":
			var req struct {
				Model    string           `json:"model"`
				Messages []ollama.Message `json:"messages"`
				Stream   bool             `json:"stream"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			if req.Stream || req.Model != "llama3.2:latest" {
				t.Errorf("bad chat request: %+v", req)
			}
			time.Sleep(50 * time.Millisecond)
			json.NewEncoder(w).Encode(map[string]any{
				"message":    map[string]string{"role": "assistant", "content": " 4 "},
				"eval_count": 20, "eval_duration": 400_000_000, "total_duration": 1_000_000_000,
			})
		case "/api/version":
			json.NewEncoder(w).Encode(map[string]string{"version": "test"})
		case "/api/show":
			json.NewEncoder(w).Encode(map[string]any{})
		case "/api/tags":
			json.NewEncoder(w).Encode(map[string]any{"models": []map[string]string{{"name": "llama3.2:latest"}}})
		}
	}))
	defer ollamaSrv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	llm := ollama.New(ollamaSrv.URL, 5*time.Second)
	model, err := firstModel(ctx, llm)
	if err != nil || model != "llama3.2:latest" {
		t.Fatalf("firstModel = %q, %v", model, err)
	}
	st, _ := store.New(t.TempDir())
	sc := sig.New(signalSrv.URL, botNum)
	b := bot.New(ctx, sc, llm, st, bot.Options{
		OwnNumber: botNum, Allowed: map[string]bool{me: true}, DefaultModel: model, HistoryLimit: 10,
	})
	done := make(chan error)
	go func() { done <- sc.Listen(ctx, b.Handle) }()

	select {
	case <-sent:
	case <-time.After(3 * time.Second):
		t.Fatal("no reply sent")
	}
	time.Sleep(100 * time.Millisecond) // let typing-stop land; make sure nothing else is sent
	cancel()
	<-done
	b.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(sends) != 1 {
		t.Fatalf("expected 1 send, got %d: %v", len(sends), sends)
	}
	s := sends[0]
	if s["number"] != botNum || s["message"] != "4" || s["recipients"].([]any)[0] != me || s["text_mode"] != "styled" {
		t.Fatalf("bad send: %v", s)
	}
	if len(typing) < 2 || typing[0] != http.MethodPut || typing[len(typing)-1] != http.MethodDelete {
		t.Fatalf("typing calls: %v", typing)
	}
	c, _ := st.Load(me)
	if len(c.History) != 2 {
		t.Fatalf("history not saved: %+v", c)
	}
	if c.Last == nil || c.Last.Stats.EvalCount != 20 || c.Last.Stats.TokensPerSecond() != 50 || c.Last.Stats.TotalDuration != time.Second {
		t.Fatalf("stats not parsed/saved: %+v", c.Last)
	}
}
