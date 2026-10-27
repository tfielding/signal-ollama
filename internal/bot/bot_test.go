package bot

import (
	"context"
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
	sent   []string
	typing []bool
	ch     chan string
}

func (f *fakeMsg) Send(_ context.Context, to, text string) error {
	f.mu.Lock()
	f.sent = append(f.sent, to+": "+text)
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
	mu    sync.Mutex
	last  []ollama.Message
	model string
}

func (f *fakeLLM) Chat(_ context.Context, model string, msgs []ollama.Message) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.last, f.model = msgs, model
	return "echo: " + msgs[len(msgs)-1].Content, nil
}
func (f *fakeLLM) Models(context.Context) ([]string, error) {
	return []string{"llama3.2:latest", "qwen2.5:7b"}, nil
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
