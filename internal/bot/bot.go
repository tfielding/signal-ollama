// Package bot routes Signal messages to Ollama and handles chat commands.
package bot

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/tfielding/signal-ollama/internal/format"
	"github.com/tfielding/signal-ollama/internal/ollama"
	"github.com/tfielding/signal-ollama/internal/signalcli"
	"github.com/tfielding/signal-ollama/internal/store"
	"github.com/tfielding/signal-ollama/internal/toolbox"
)

// Messenger is the Signal side (implemented by signalcli.Client).
type Messenger interface {
	Send(ctx context.Context, recipient, text string, styled bool, attachments []signalcli.Attachment) error
	Typing(ctx context.Context, recipient string, on bool) error
}

// LLM is the model side (implemented by ollama.Client).
type LLM interface {
	Chat(ctx context.Context, model string, msgs []ollama.Message, think any, tools []ollama.Tool) (ollama.Result, error)
	Models(ctx context.Context) ([]string, error)
	Ping(ctx context.Context) error
	Info(ctx context.Context, model string) (ollama.ModelInfo, error)
}

// Tools are the MCP tools the model may use (implemented by toolbox.Box).
type Tools interface {
	Tools(ctx context.Context) []toolbox.Tool
	Call(ctx context.Context, name string, args map[string]any) (toolbox.Output, error)
	Errors() map[string]error
	Guidance(tools []toolbox.Tool) string
}

type Options struct {
	Tools         Tools // nil = no tools
	OwnNumber     string
	Allowed       map[string]bool
	DefaultModel  string
	SystemPrompt  string
	HistoryLimit  int
	TypingRefresh time.Duration // Signal clients hide the indicator after ~15s
}

type Bot struct {
	msg   Messenger
	llm   LLM
	store *store.Store
	opt   Options

	mu      sync.Mutex
	queues  map[string]chan string // one worker per sender keeps replies in order
	active  map[string]activity    // model call currently running, per sender
	workers sync.WaitGroup
}

type activity struct {
	model   string
	started time.Time
	tool    string             // tool currently running, if any
	cancel  context.CancelFunc // stops this model call (/stop)
}

func New(m Messenger, l LLM, s *store.Store, opt Options) *Bot {
	if opt.TypingRefresh == 0 {
		opt.TypingRefresh = 10 * time.Second
	}
	return &Bot{msg: m, llm: l, store: s, opt: opt,
		queues: map[string]chan string{}, active: map[string]activity{}}
}

// Handle is called for every incoming envelope. It never blocks on the model.
// ctx governs the whole bot: when it is cancelled, workers stop.
func (b *Bot) Handle(ctx context.Context, env signalcli.Envelope) {
	sender, text, ok := b.accept(env)
	if !ok {
		return
	}
	// Read-only commands skip the queue so they answer even mid-reply.
	if instant(text) {
		b.workers.Add(1)
		go func() {
			defer b.workers.Done()
			b.process(ctx, sender, text, 0)
		}()
		return
	}
	b.mu.Lock()
	q, exists := b.queues[sender]
	if !exists {
		q = make(chan string, 16)
		b.queues[sender] = q
		b.workers.Add(1)
		go b.worker(ctx, sender, q)
	}
	b.mu.Unlock()

	select {
	case q <- text:
	default:
		slog.Warn("queue full, dropping message", "sender", sender)
	}
}

// Wait blocks until all workers have exited (after ctx is cancelled).
func (b *Bot) Wait() { b.workers.Wait() }

// accept applies the boundaries: direct text messages from allowed senders only.
// Messages in group chats, sync messages (our own
// outgoing alerts), receipts and typing events are all ignored.
func (b *Bot) accept(env signalcli.Envelope) (sender, text string, ok bool) {
	dm := env.DataMessage
	if dm == nil || dm.GroupInfo != nil {
		return "", "", false
	}
	text = strings.TrimSpace(dm.Message)
	if text == "" {
		return "", "", false
	}
	sender = env.SourceNumber
	if sender == "" {
		sender = env.Source
	}
	if sender == "" || sender == b.opt.OwnNumber {
		return "", "", false
	}
	if !b.opt.Allowed[sender] && !b.opt.Allowed[env.SourceUUID] {
		slog.Info("ignoring message from sender not on allowlist", "from", mask(sender))
		return "", "", false
	}
	return sender, text, true
}

func (b *Bot) worker(ctx context.Context, sender string, q chan string) {
	defer b.workers.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case text := <-q:
			b.process(ctx, sender, text, len(q))
		}
	}
}

// process answers one message and sends the reply.
func (b *Bot) process(ctx context.Context, sender, text string, queued int) {
	start := time.Now()
	kind := "chat"
	if strings.HasPrefix(text, "/") {
		kind = "command"
	}
	slog.Info("message received", "from", mask(sender), "kind", kind, "chars", len(text), "queued", queued)
	r := b.respond(ctx, sender, text)
	if r.text == "" && len(r.images) == 0 {
		return
	}
	if err := b.send(ctx, sender, r); err != nil {
		slog.Error("send failed", "to", mask(sender), "err", err)
		return
	}
	slog.Info("reply sent", "to", mask(sender), "chars", len(r.text), "images", len(r.images), "took", time.Since(start).Round(100*time.Millisecond))
}

// send delivers a reply. Model replies are Markdown, converted to Signal formatting;
// if the styled send fails, the raw text is sent as a fallback.
func (b *Bot) send(ctx context.Context, to string, r response) error {
	var att []signalcli.Attachment
	for i, img := range r.images {
		att = append(att, signalcli.Attachment{MIME: img.MIME, Name: fmt.Sprintf("image-%d%s", i+1, ext(img.MIME)), Data: img.Data})
	}
	if !r.fromModel {
		return b.msg.Send(ctx, to, r.text, false, att)
	}
	if err := b.msg.Send(ctx, to, format.ToSignal(r.text), true, att); err != nil {
		slog.Warn("styled send failed, retrying as plain text", "err", err)
		return b.msg.Send(ctx, to, r.text, false, att)
	}
	return nil
}

func ext(mime string) string {
	switch mime {
	case "image/jpeg":
		return ".jpg"
	case "image/png":
		return ".png"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	}
	return ""
}

// mask hides all but the last 4 characters of a phone number in logs.
func mask(s string) string {
	if len(s) <= 4 {
		return s
	}
	return "…" + s[len(s)-4:]
}
