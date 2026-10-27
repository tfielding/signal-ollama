package ollama

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestThinkAndChat(t *testing.T) {
	var gotThink any = "unset"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/version":
			w.Write([]byte(`{"version":"0.12.0"}`))
		case "/api/show":
			var req map[string]string
			json.NewDecoder(r.Body).Decode(&req)
			switch req["model"] {
			case "gpt-oss":
				w.Write([]byte(`{"thinking":{"values":["low","medium","high"],"default":"medium"}}`))
			case "qwen":
				w.Write([]byte(`{"thinking":{"values":[true,false],"default":true}}`))
			default:
				w.Write([]byte(`{"details":{}}`))
			}
		case "/api/chat":
			var req map[string]any
			json.NewDecoder(r.Body).Decode(&req)
			gotThink = "absent"
			if v, ok := req["think"]; ok {
				gotThink = v
			}
			w.Write([]byte(`{"message":{"role":"assistant","content":"ok","thinking":"hmm"},"eval_count":10,"eval_duration":1000000000}`))
		}
	}))
	defer srv.Close()
	c := New(srv.URL, 5*time.Second)
	ctx := context.Background()

	if err := c.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	gi, _ := c.Info(ctx, "gpt-oss")
	g := gi.Think
	if !g.Known || g.Default != "medium" || !g.Supports("low") || g.Supports("off") {
		t.Fatalf("gpt-oss: %+v", g)
	}
	qi, _ := c.Info(ctx, "qwen")
	q := qi.Think
	if !q.Known || q.Default != "on" || !q.Supports("off") {
		t.Fatalf("qwen: %+v", q)
	}
	if u, _ := c.Info(ctx, "llama"); u.Think.Known {
		t.Fatalf("llama: %+v", u)
	}

	for _, tc := range []struct {
		setting string
		want    any
	}{{"", "absent"}, {"off", false}, {"on", true}, {"low", "low"}} {
		res, err := c.Chat(ctx, "m", nil, ThinkValue(tc.setting), nil)
		if err != nil || res.Content != "ok" || res.Stats.TokensPerSecond() != 10 {
			t.Fatalf("chat: %+v %v", res, err)
		}
		if gotThink != tc.want {
			t.Fatalf("setting %q sent think=%#v, want %#v", tc.setting, gotThink, tc.want)
		}
	}
}

func TestPingFailsFast(t *testing.T) {
	c := New("http://127.0.0.1:1", time.Minute) // nothing listens on port 1
	start := time.Now()
	if err := c.Ping(context.Background()); err == nil {
		t.Fatal("expected error")
	}
	if time.Since(start) > 4*time.Second {
		t.Fatal("ping too slow")
	}
}

func TestChatCancel(t *testing.T) {
	aborted := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)   // like Ollama, read the request first
		<-r.Context().Done() // a slow generation; ends when the client goes away
		close(aborted)
	}))
	defer srv.Close()
	c := New(srv.URL, time.Minute)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	start := time.Now()
	if _, err := c.Chat(ctx, "m", nil, nil, nil); err == nil {
		t.Fatal("expected error")
	}
	if time.Since(start) > 2*time.Second {
		t.Fatal("cancel did not abort the request")
	}
	select {
	case <-aborted:
	case <-time.After(2 * time.Second):
		t.Fatal("server never saw the request end")
	}
}
