package bot

import (
	"context"
	"errors"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"signal-ollama/internal/ollama"
	"signal-ollama/internal/signal"
	"signal-ollama/internal/store"
)

const (
	me    = "+15550000001" // allowed user
	own   = "+15550000009" // the bot's number
	other = "+15550000002" // someone not on the allowlist
)

type fakeMsg struct {
	mu     sync.Mutex
	styled []bool
	sent   []string
	typing []bool
	ch     chan string
}

func (f *fakeMsg) Send(_ context.Context, to, text string, styled bool) error {
	f.mu.Lock()
	f.sent = append(f.sent, to+": "+text)
	f.styled = append(f.styled, styled)
	f.mu.Unlock()
	f.ch <- text
	return nil
}
func (f *fakeMsg) Typing(_ context.Context, _ string, on bool) error {
	f.mu.Lock()
	f.typing = append(f.typing, on)
	f.mu.Unlock()
	return nil
}

type fakeLLM struct {
	mu      sync.Mutex
	last    []ollama.Message
	model   string
	think   any
	calls   int
	gate    chan struct{} // if set, Chat blocks until it's closed
	pingErr error
	chatErr error
}

func (f *fakeLLM) Chat(_ context.Context, model string, msgs []ollama.Message, think any) (ollama.Result, error) {
	f.mu.Lock()
	f.last, f.model, f.think = msgs, model, think
	f.calls++
	gate, chatErr := f.gate, f.chatErr
	f.mu.Unlock()
	if gate != nil {
		<-gate
	}
	if chatErr != nil {
		return ollama.Result{}, chatErr
	}
	return ollama.Result{
		Content: "echo: " + msgs[len(msgs)-1].Content,
		Stats:   ollama.Stats{EvalCount: 100, EvalDuration: 2 * time.Second, PromptEvalCount: 50, PromptEvalDuration: 500 * time.Millisecond, TotalDuration: 3 * time.Second},
	}, nil
}
func (f *fakeLLM) Models(context.Context) ([]string, error) {
	return []string{"llama3.2:latest", "qwen2.5:7b", "gpt-oss:20b"}, nil
}
func (f *fakeLLM) Ping(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.pingErr
}
func (f *fakeLLM) Think(_ context.Context, model string) (ollama.ThinkInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pingErr != nil {
		return ollama.ThinkInfo{}, f.pingErr
	}
	switch model {
	case "qwen2.5:7b":
		return ollama.ThinkInfo{Known: true, Values: []string{"on", "off"}, Default: "on"}, nil
	case "gpt-oss:20b":
		return ollama.ThinkInfo{Known: true, Values: []string{"low", "medium", "high"}, Default: "medium"}, nil
	}
	return ollama.ThinkInfo{}, nil // no metadata
}

func setup(t *testing.T) (*Bot, *fakeMsg, *fakeLLM, *store.Store) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	st, err := store.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	m := &fakeMsg{ch: make(chan string, 10)}
	l := &fakeLLM{}
	b := New(ctx, m, l, st, Options{
		OwnNumber: own, Allowed: map[string]bool{me: true}, DefaultModel: "llama3.2:latest",
		SystemPrompt: "sys", HistoryLimit: 4, TypingRefresh: 5 * time.Millisecond,
	})
	t.Cleanup(func() { cancel(); b.Wait() })
	return b, m, l, st
}

func dm(from, text string) signal.Envelope {
	return signal.Envelope{SourceNumber: from, Source: from, DataMessage: &signal.DataMessage{Message: text}}
}

func reply(t *testing.T, m *fakeMsg) string {
	t.Helper()
	select {
	case r := <-m.ch:
		return r
	case <-time.After(2 * time.Second):
		t.Fatal("no reply")
		return ""
	}
}

func expectSilence(t *testing.T, m *fakeMsg) {
	t.Helper()
	select {
	case r := <-m.ch:
		t.Fatalf("unexpected reply %q", r)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestBoundaries(t *testing.T) {
	b, m, _, _ := setup(t)

	group := dm(me, "hi in a group chat")
	group.DataMessage.GroupInfo = &signal.GroupInfo{GroupID: "abc"}
	b.Handle(group)
	b.Handle(dm(other, "hi from someone else"))
	b.Handle(dm(own, "outgoing alert"))
	b.Handle(signal.Envelope{SourceNumber: own, SyncMessage: []byte(`{"sentMessage":{"message":"alert"}}`)})
	b.Handle(signal.Envelope{SourceNumber: me}) // receipt / typing, no dataMessage
	b.Handle(dm(me, "   "))
	expectSilence(t, m)

	b.Handle(dm(me, "hello"))
	if r := reply(t, m); r != "echo: hello" {
		t.Fatalf("got %q", r)
	}
	if !strings.HasPrefix(m.sent[0], me+": ") {
		t.Fatalf("replied to wrong recipient: %q", m.sent[0])
	}
}

func TestAllowByUUID(t *testing.T) {
	b, m, _, _ := setup(t)
	b.opt.Allowed["uuid-123"] = true
	env := dm(other, "hi")
	env.SourceUUID = "uuid-123"
	b.Handle(env)
	if r := reply(t, m); r != "echo: hi" {
		t.Fatalf("got %q", r)
	}
}

func TestHistoryPersistsAndTrims(t *testing.T) {
	b, m, l, st := setup(t)
	for _, s := range []string{"one", "two", "three"} {
		b.Handle(dm(me, s))
		reply(t, m)
	}
	// system + last 4 history msgs (two, echo two... trimmed to limit 4) + new user msg
	l.mu.Lock()
	got := l.last
	l.mu.Unlock()
	if got[0].Role != "system" || got[1].Content != "one" || got[len(got)-1].Content != "three" {
		t.Fatalf("unexpected prompt: %+v", got)
	}
	chat, _ := st.Load(me)
	if len(chat.History) != 4 || chat.History[0].Content != "two" || chat.History[0].Role != "user" {
		t.Fatalf("history not trimmed: %+v", chat.History)
	}

	b.Handle(dm(me, "/reset"))
	if r := reply(t, m); r != "History cleared." {
		t.Fatalf("got %q", r)
	}
	chat, _ = st.Load(me)
	if len(chat.History) != 0 {
		t.Fatal("history not cleared")
	}
}

func TestModelCommands(t *testing.T) {
	b, m, l, st := setup(t)

	b.Handle(dm(me, "/model"))
	if r := reply(t, m); !strings.Contains(r, "* llama3.2:latest") || !strings.Contains(r, "qwen2.5:7b") {
		t.Fatalf("list: %q", r)
	}
	b.Handle(dm(me, "/model nope"))
	if r := reply(t, m); !strings.Contains(r, "No model") {
		t.Fatalf("bad model: %q", r)
	}
	b.Handle(dm(me, "/model qwen2.5:7b"))
	if r := reply(t, m); r != "Switched to qwen2.5:7b." {
		t.Fatalf("switch: %q", r)
	}
	b.Handle(dm(me, "hi"))
	reply(t, m)
	if l.model != "qwen2.5:7b" {
		t.Fatalf("used model %q", l.model)
	}
	if c, _ := st.Load(me); c.Model != "qwen2.5:7b" {
		t.Fatal("model not persisted")
	}
	b.Handle(dm(me, "/model llama3.2")) // without tag
	if r := reply(t, m); r != "Switched to llama3.2:latest." {
		t.Fatalf("tagless: %q", r)
	}
	b.Handle(dm(me, "/status"))
	if r := reply(t, m); !strings.Contains(r, "llama3.2:latest") {
		t.Fatalf("status: %q", r)
	}
}

func TestTypingIndicator(t *testing.T) {
	b, m, _, _ := setup(t)
	b.Handle(dm(me, "hello"))
	reply(t, m)
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.typing) < 2 || !m.typing[0] || m.typing[len(m.typing)-1] {
		t.Fatalf("typing sequence: %v", m.typing)
	}
}

func TestStatusAnswersMidReply(t *testing.T) {
	b, m, l, _ := setup(t)
	b.Handle(dm(me, "/status"))
	if r := reply(t, m); !strings.Contains(r, "Now: idle") {
		t.Fatalf("idle status: %q", r)
	}

	gate := make(chan struct{})
	l.mu.Lock()
	l.gate = gate
	l.mu.Unlock()
	b.Handle(dm(me, "slow question"))
	b.Handle(dm(me, "second question"))
	// wait until the model call has started
	deadline := time.Now().Add(2 * time.Second)
	for {
		b.mu.Lock()
		_, busy := b.active[me]
		b.mu.Unlock()
		if busy || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	b.Handle(dm(me, "/status"))
	r := reply(t, m) // must arrive while the model is still blocked
	if !strings.Contains(r, "Now: llama3.2:latest thinking for") || !strings.Contains(r, "1 message(s) waiting") {
		t.Fatalf("busy status: %q", r)
	}
	close(gate)
	if r := reply(t, m); r != "echo: slow question" {
		t.Fatalf("got %q", r)
	}
	reply(t, m)
}

func TestStats(t *testing.T) {
	b, m, _, _ := setup(t)
	b.Handle(dm(me, "/stats"))
	if r := reply(t, m); r != "No replies yet." {
		t.Fatalf("got %q", r)
	}
	b.Handle(dm(me, "hello"))
	reply(t, m)
	b.Handle(dm(me, "/stats"))
	r := reply(t, m)
	for _, want := range []string{"Last reply: llama3.2:latest", "Output: 100 tokens at 50.0 tok/s (2s)", "Prompt: 50 tokens at 100 tok/s", "Total: 3s"} {
		if !strings.Contains(r, want) {
			t.Fatalf("missing %q in %q", want, r)
		}
	}
}

func TestModelRepliesStyledCommandsPlain(t *testing.T) {
	b, m, _, _ := setup(t)
	b.Handle(dm(me, "## Title"))
	if r := reply(t, m); r != "echo: ## Title" { // fake echoes; "##" is mid-line so not a heading
		t.Fatalf("got %q", r)
	}
	b.Handle(dm(me, "/help"))
	reply(t, m)
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.styled[0] || m.styled[1] {
		t.Fatalf("styled flags: %v", m.styled)
	}
}

var refused = &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connect: connection refused")}

func TestOllamaOffline(t *testing.T) {
	b, m, l, st := setup(t)
	l.pingErr = refused
	b.Handle(dm(me, "hello"))
	if r := reply(t, m); r != offlineText {
		t.Fatalf("got %q", r)
	}
	if l.calls != 0 {
		t.Fatal("should not call the model when offline")
	}
	if c, _ := st.Load(me); len(c.History) != 0 {
		t.Fatal("offline message should not be saved")
	}
	b.Handle(dm(me, "/status"))
	if r := reply(t, m); !strings.Contains(r, "Ollama: offline (connect: connection refused)") {
		t.Fatalf("status: %q", r)
	}

	// Goes away mid-reply.
	l.mu.Lock()
	l.pingErr, l.chatErr = nil, refused
	l.mu.Unlock()
	b.Handle(dm(me, "hello"))
	if r := reply(t, m); r != offlineText {
		t.Fatalf("mid-reply: %q", r)
	}
}

func TestThink(t *testing.T) {
	b, m, l, st := setup(t)
	send := func(text string) string { b.Handle(dm(me, text)); return reply(t, m) }

	// Default model has no thinking metadata: generic options.
	if r := send("/think"); !strings.Contains(r, "Thinking: model default") || !strings.Contains(r, "on, off, default") {
		t.Fatalf("think (unknown): %q", r)
	}
	send("/model qwen2.5:7b")
	if r := send("/think"); !strings.Contains(r, "model default (on)") || !strings.Contains(r, "Options for qwen2.5:7b: on, off, default") {
		t.Fatalf("think (qwen): %q", r)
	}
	if r := send("/think off"); r != "Thinking: off." {
		t.Fatalf("set off: %q", r)
	}
	send("hi")
	if l.think != false {
		t.Fatalf("think sent = %#v", l.think)
	}
	if c, _ := st.Load(me); c.Think != "off" {
		t.Fatal("think not persisted")
	}

	// gpt-oss only has levels: switching warns, and the unsupported setting isn't sent.
	if r := send("/model gpt-oss:20b"); !strings.Contains(r, `doesn't support thinking "off"`) {
		t.Fatalf("switch warning: %q", r)
	}
	send("hi")
	if l.think != nil {
		t.Fatalf("unsupported setting sent: %#v", l.think)
	}
	if r := send("/think off"); !strings.Contains(r, `doesn't support "off". Options: low, medium, high, default`) {
		t.Fatalf("reject off: %q", r)
	}
	if r := send("/think low"); r != "Thinking: low." {
		t.Fatalf("set low: %q", r)
	}
	send("hi")
	if l.think != "low" {
		t.Fatalf("think sent = %#v", l.think)
	}
	if r := send("/status"); !strings.Contains(r, "Thinking: low") || !strings.Contains(r, "Ollama: online") {
		t.Fatalf("status: %q", r)
	}
	if r := send("/think default"); r != "Thinking: model default (medium)." {
		t.Fatalf("default: %q", r)
	}
	send("hi")
	if l.think != nil {
		t.Fatalf("default sent %#v", l.think)
	}
}
