// Package bot routes Signal messages to Ollama and handles chat commands.
package bot

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"signal-ollama/internal/ollama"
	"signal-ollama/internal/signal"
	"signal-ollama/internal/store"
)

// Messenger is the Signal side (implemented by signal.Client).
type Messenger interface {
	Send(ctx context.Context, recipient, text string) error
	Typing(ctx context.Context, recipient string, on bool) error
}

// LLM is the model side (implemented by ollama.Client).
type LLM interface {
	Chat(ctx context.Context, model string, msgs []ollama.Message) (string, error)
	Models(ctx context.Context) ([]string, error)
}

type Options struct {
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

	ctx     context.Context
	mu      sync.Mutex
	queues  map[string]chan string // one worker per sender keeps replies in order
	workers sync.WaitGroup
}

func New(ctx context.Context, m Messenger, l LLM, s *store.Store, opt Options) *Bot {
	if opt.TypingRefresh == 0 {
		opt.TypingRefresh = 10 * time.Second
	}
	return &Bot{msg: m, llm: l, store: s, opt: opt, ctx: ctx, queues: map[string]chan string{}}
}

// Handle is called for every incoming envelope. It never blocks on the model.
func (b *Bot) Handle(env signal.Envelope) {
	sender, text, ok := b.accept(env)
	if !ok {
		return
	}
	b.mu.Lock()
	q, exists := b.queues[sender]
	if !exists {
		q = make(chan string, 16)
		b.queues[sender] = q
		b.workers.Add(1)
		go b.worker(sender, q)
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
// Group messages (e.g. the family security group), sync messages (our own
// outgoing alerts), receipts and typing events are all ignored.
func (b *Bot) accept(env signal.Envelope) (sender, text string, ok bool) {
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
		slog.Info("ignoring message from sender not on allowlist", "sender", sender)
		return "", "", false
	}
	return sender, text, true
}

func (b *Bot) worker(sender string, q chan string) {
	defer b.workers.Done()
	for {
		select {
		case <-b.ctx.Done():
			return
		case text := <-q:
			reply := b.respond(sender, text)
			if reply == "" {
				continue
			}
			if err := b.msg.Send(b.ctx, sender, reply); err != nil {
				slog.Error("send failed", "err", err)
			}
		}
	}
}

func (b *Bot) respond(sender, text string) string {
	chat, err := b.store.Load(sender)
	if err != nil {
		slog.Error("load history", "err", err)
		return "Couldn't load chat history: " + err.Error()
	}
	if strings.HasPrefix(text, "/") {
		return b.command(sender, &chat, text)
	}

	model := chat.Model
	if model == "" {
		model = b.opt.DefaultModel
	}
	msgs := make([]ollama.Message, 0, len(chat.History)+2)
	if b.opt.SystemPrompt != "" {
		msgs = append(msgs, ollama.Message{Role: "system", Content: b.opt.SystemPrompt})
	}
	msgs = append(msgs, chat.History...)
	msgs = append(msgs, ollama.Message{Role: "user", Content: text})

	stopTyping := b.typing(sender)
	reply, err := b.llm.Chat(b.ctx, model, msgs)
	stopTyping()
	if err != nil {
		slog.Error("ollama chat", "model", model, "err", err)
		return "Model error: " + err.Error()
	}
	if reply == "" {
		reply = "(empty reply)"
	}

	chat.History = trim(append(chat.History,
		ollama.Message{Role: "user", Content: text},
		ollama.Message{Role: "assistant", Content: reply}), b.opt.HistoryLimit)
	if err := b.store.Save(sender, chat); err != nil {
		slog.Error("save history", "err", err)
	}
	return reply
}

// typing shows the indicator until the returned func is called.
func (b *Bot) typing(to string) (stop func()) {
	done := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(b.opt.TypingRefresh)
		defer t.Stop()
		for {
			if err := b.msg.Typing(b.ctx, to, true); err != nil {
				slog.Warn("typing indicator failed", "err", err)
			}
			select {
			case <-done:
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				_ = b.msg.Typing(ctx, to, false)
				cancel()
				return
			case <-b.ctx.Done():
				return
			case <-t.C:
			}
		}
	}()
	return func() { close(done); <-finished }
}

func (b *Bot) command(sender string, chat *store.Chat, text string) string {
	fields := strings.Fields(text)
	cmd, args := strings.ToLower(fields[0]), fields[1:]
	current := chat.Model
	if current == "" {
		current = b.opt.DefaultModel
	}

	switch cmd {
	case "/help":
		return helpText
	case "/reset":
		chat.History = nil
		if err := b.store.Save(sender, *chat); err != nil {
			return "Couldn't reset: " + err.Error()
		}
		return "History cleared."
	case "/status":
		return fmt.Sprintf("Model: %s\nMessages in memory: %d (max %d)", current, len(chat.History), b.opt.HistoryLimit)
	case "/model":
		models, err := b.llm.Models(b.ctx)
		if err != nil {
			return "Couldn't list models: " + err.Error()
		}
		if len(args) == 0 {
			var sb strings.Builder
			sb.WriteString("Models (* = current):\n")
			for _, m := range models {
				mark := "  "
				if m == current {
					mark = "* "
				}
				sb.WriteString(mark + m + "\n")
			}
			sb.WriteString("\nSwitch with /model <name>")
			return sb.String()
		}
		name, ok := match(models, args[0])
		if !ok {
			return fmt.Sprintf("No model named %q. Send /model to see the list.", args[0])
		}
		chat.Model = name
		if name == b.opt.DefaultModel {
			chat.Model = ""
		}
		if err := b.store.Save(sender, *chat); err != nil {
			return "Couldn't save: " + err.Error()
		}
		return "Switched to " + name + "."
	default:
		return "Unknown command. " + helpText
	}
}

const helpText = `Commands:
/model - list models
/model <name> - switch model
/reset - clear conversation memory
/status - current model and memory
/help - this message`

// match finds a model by exact name, or by name without the ":tag" (e.g. "llama3.2" -> "llama3.2:latest").
func match(models []string, want string) (string, bool) {
	for _, m := range models {
		if m == want {
			return m, true
		}
	}
	for _, m := range models {
		if base, _, _ := strings.Cut(m, ":"); base == want {
			return m, true
		}
	}
	return "", false
}

// trim keeps the newest limit messages, starting on a user turn.
func trim(h []ollama.Message, limit int) []ollama.Message {
	if len(h) <= limit {
		return h
	}
	h = h[len(h)-limit:]
	for len(h) > 0 && h[0].Role != "user" {
		h = h[1:]
	}
	return append([]ollama.Message(nil), h...)
}
