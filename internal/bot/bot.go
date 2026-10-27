// Package bot routes Signal messages to Ollama and handles chat commands.
package bot

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strings"
	"sync"
	"time"

	"signal-ollama/internal/format"
	"signal-ollama/internal/ollama"
	"signal-ollama/internal/signal"
	"signal-ollama/internal/store"
)

// Messenger is the Signal side (implemented by signal.Client).
type Messenger interface {
	Send(ctx context.Context, recipient, text string, styled bool) error
	Typing(ctx context.Context, recipient string, on bool) error
}

// LLM is the model side (implemented by ollama.Client).
type LLM interface {
	Chat(ctx context.Context, model string, msgs []ollama.Message, think any) (ollama.Result, error)
	Models(ctx context.Context) ([]string, error)
	Ping(ctx context.Context) error
	Think(ctx context.Context, model string) (ollama.ThinkInfo, error)
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
	active  map[string]activity    // model call currently running, per sender
	workers sync.WaitGroup
}

type activity struct {
	model   string
	started time.Time
	cancel  context.CancelFunc // stops this model call (/stop)
}

func New(ctx context.Context, m Messenger, l LLM, s *store.Store, opt Options) *Bot {
	if opt.TypingRefresh == 0 {
		opt.TypingRefresh = 10 * time.Second
	}
	return &Bot{msg: m, llm: l, store: s, opt: opt, ctx: ctx,
		queues: map[string]chan string{}, active: map[string]activity{}}
}

// Handle is called for every incoming envelope. It never blocks on the model.
func (b *Bot) Handle(env signal.Envelope) {
	sender, text, ok := b.accept(env)
	if !ok {
		return
	}
	// Read-only commands skip the queue so they answer even mid-reply.
	if instant(text) {
		b.workers.Add(1)
		go func() {
			defer b.workers.Done()
			b.process(sender, text, 0)
		}()
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
// Messages in group chats, sync messages (our own
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
		slog.Info("ignoring message from sender not on allowlist", "from", mask(sender))
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
			b.process(sender, text, len(q))
		}
	}
}

// process answers one message and sends the reply.
func (b *Bot) process(sender, text string, queued int) {
	start := time.Now()
	kind := "chat"
	if strings.HasPrefix(text, "/") {
		kind = "command"
	}
	slog.Info("message received", "from", mask(sender), "kind", kind, "chars", len(text), "queued", queued)
	reply, fromModel := b.respond(sender, text)
	if reply == "" {
		return
	}
	if err := b.send(sender, reply, fromModel); err != nil {
		slog.Error("send failed", "to", mask(sender), "err", err)
		return
	}
	slog.Info("reply sent", "to", mask(sender), "chars", len(reply), "took", time.Since(start).Round(100*time.Millisecond))
}

// send delivers a reply. Model replies are Markdown, converted to Signal formatting;
// if the styled send fails, the raw text is sent as a fallback.
func (b *Bot) send(to, reply string, fromModel bool) error {
	if !fromModel {
		return b.msg.Send(b.ctx, to, reply, false)
	}
	if err := b.msg.Send(b.ctx, to, format.ToSignal(reply), true); err != nil {
		slog.Warn("styled send failed, retrying as plain text", "err", err)
		return b.msg.Send(b.ctx, to, reply, false)
	}
	return nil
}

func instant(text string) bool {
	f := strings.Fields(strings.ToLower(text))
	return len(f) > 0 && (f[0] == "/status" || f[0] == "/stats" || f[0] == "/stop")
}

// respond returns the reply and whether it came from the model (Markdown) rather than a command.
func (b *Bot) respond(sender, text string) (string, bool) {
	chat, err := b.store.Load(sender)
	if err != nil {
		slog.Error("load history", "err", err)
		return "Couldn't load chat history: " + err.Error(), false
	}
	if strings.HasPrefix(text, "/") {
		return b.command(sender, &chat, text), false
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

	// Fail fast with a clear message if Ollama is down (e.g. the machine is asleep).
	if err := b.llm.Ping(b.ctx); err != nil {
		slog.Warn("ollama unreachable", "err", err)
		return offlineText, false
	}
	think := b.thinkValue(model, chat.Think)

	slog.Info("asking model", "model", model, "history", len(chat.History), "think", chat.Think)
	start := time.Now()
	b.mu.Lock()
	callCtx, cancel := context.WithCancel(b.ctx)
	defer cancel()
	b.active[sender] = activity{model: model, started: start, cancel: cancel}
	b.mu.Unlock()
	stopTyping := b.typing(sender)
	res, err := b.llm.Chat(callCtx, model, msgs, think)
	stopTyping()
	b.mu.Lock()
	delete(b.active, sender)
	b.mu.Unlock()
	if err != nil && callCtx.Err() != nil && b.ctx.Err() == nil {
		// Cancelled by /stop, which already replied. Nothing is saved.
		slog.Info("model call stopped", "model", model, "took", time.Since(start).Round(100*time.Millisecond))
		return "", false
	}
	if err != nil {
		slog.Error("ollama chat", "model", model, "took", time.Since(start).Round(100*time.Millisecond), "err", err)
		if unreachable(err) {
			return offlineText, false
		}
		return "Model error: " + err.Error(), false
	}
	slog.Info("model replied", "model", model, "took", time.Since(start).Round(100*time.Millisecond),
		"tokens", res.Stats.EvalCount, "tok_per_s", fmt.Sprintf("%.1f", res.Stats.TokensPerSecond()))
	reply := res.Content
	if reply == "" {
		reply = "(empty reply)"
	}

	chat.Last = &store.LastReply{Model: model, At: time.Now(), Stats: res.Stats}
	chat.History = trim(append(chat.History,
		ollama.Message{Role: "user", Content: text},
		ollama.Message{Role: "assistant", Content: reply}), b.opt.HistoryLimit)
	if err := b.store.Save(sender, chat); err != nil {
		slog.Error("save history", "err", err)
	}
	return reply, true
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
		ollamaState := "online"
		thinking := "unknown (Ollama offline)"
		if err := b.llm.Ping(b.ctx); err != nil {
			ollamaState = "offline (" + short(err) + ")"
		} else if info, err := b.llm.Think(b.ctx, current); err == nil {
			thinking = thinkLabel(chat.Think, info)
		}
		return fmt.Sprintf("Model: %s\nThinking: %s\nMemory: %d messages (max %d)\nOllama: %s\nNow: %s",
			current, thinking, len(chat.History), b.opt.HistoryLimit, ollamaState, b.now(sender))
	case "/think":
		return b.thinkCommand(sender, chat, current, args)
	case "/stop":
		return b.stop(sender)
	case "/stats":
		return statsText(chat.Last)
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
		msg := "Switched to " + name + "."
		if chat.Think != "" {
			if info, err := b.llm.Think(b.ctx, name); err == nil && info.Known && !info.Supports(chat.Think) {
				msg += fmt.Sprintf("\nIt doesn't support thinking %q, so it will use its default. Options: %s", chat.Think, options(info))
			}
		}
		return msg
	default:
		return "Unknown command. " + helpText
	}
}

const helpText = `Commands:
/model - list models
/model <name> - switch model
/reset - clear conversation memory
/status - model, memory, and what's running now
/think - show or change reasoning (on, off, low, medium, high, default)
/stop - cancel the reply in progress (and anything waiting)
/stats - speed of the last reply
/help - this message`

const offlineText = "Can't reach Ollama right now. The machine running it may be asleep or offline. Try again in a bit."

// unreachable reports whether err means Ollama couldn't be contacted (as opposed to a slow or failed reply).
func unreachable(err error) bool {
	var oe *net.OpError
	return errors.As(err, &oe) && !oe.Timeout()
}

// stop cancels the model call in progress for sender and drops queued messages.
func (b *Bot) stop(sender string) string {
	b.mu.Lock()
	a, busy := b.active[sender]
	q := b.queues[sender]
	b.mu.Unlock()

	dropped := 0
	if q != nil {
	drain:
		for {
			select {
			case <-q:
				dropped++
			default:
				break drain
			}
		}
	}
	if busy {
		a.cancel()
	}

	switch {
	case busy && dropped > 0:
		return fmt.Sprintf("Stopped, and dropped %d waiting message(s).", dropped)
	case busy:
		return "Stopped."
	case dropped > 0:
		return fmt.Sprintf("Dropped %d waiting message(s).", dropped)
	}
	return "Nothing is running."
}

func short(err error) string {
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return "timed out"
	}
	var oe *net.OpError
	if errors.As(err, &oe) && oe.Err != nil {
		return oe.Err.Error()
	}
	return err.Error()
}

// thinkValue returns the API value for a chat's thinking setting, or nil (model default)
// if it's unset or the model doesn't support it.
func (b *Bot) thinkValue(model, setting string) any {
	if setting == "" {
		return nil
	}
	if info, err := b.llm.Think(b.ctx, model); err == nil && info.Known && !info.Supports(setting) {
		slog.Info("thinking setting not supported by model, using default", "model", model, "think", setting)
		return nil
	}
	return ollama.ThinkValue(setting)
}

func (b *Bot) thinkCommand(sender string, chat *store.Chat, model string, args []string) string {
	info, err := b.llm.Think(b.ctx, model)
	if err != nil {
		if unreachable(err) {
			return offlineText
		}
		return "Couldn't check the model: " + err.Error()
	}
	if len(args) == 0 {
		return fmt.Sprintf("Thinking: %s\nOptions for %s: %s\nChange with /think <option>", thinkLabel(chat.Think, info), model, options(info))
	}

	v := strings.ToLower(args[0])
	switch {
	case v == "default":
		v = ""
	case info.Known && !info.Supports(v):
		if len(info.Values) == 1 && info.Values[0] == "off" {
			return model + " doesn't support thinking."
		}
		return fmt.Sprintf("%s doesn't support %q. Options: %s", model, v, options(info))
	case !info.Known && v != "on" && v != "off" && v != "low" && v != "medium" && v != "high":
		return fmt.Sprintf("Unknown option %q. Options: %s", v, options(info))
	}
	chat.Think = v
	if err := b.store.Save(sender, *chat); err != nil {
		return "Couldn't save: " + err.Error()
	}
	return "Thinking: " + thinkLabel(v, info) + "."
}

func options(info ollama.ThinkInfo) string {
	if !info.Known {
		return "on, off, default"
	}
	return strings.Join(info.Values, ", ") + ", default"
}

func thinkLabel(setting string, info ollama.ThinkInfo) string {
	if setting == "" {
		if info.Default != "" {
			return "model default (" + info.Default + ")"
		}
		return "model default"
	}
	if info.Known && !info.Supports(setting) {
		return setting + " (not supported by this model, using its default)"
	}
	return setting
}

// now describes what the bot is doing for sender right now.
func (b *Bot) now(sender string) string {
	b.mu.Lock()
	a, busy := b.active[sender]
	waiting := 0
	if q := b.queues[sender]; q != nil {
		waiting = len(q)
	}
	b.mu.Unlock()

	s := "idle"
	if busy {
		s = fmt.Sprintf("%s thinking for %s", a.model, time.Since(a.started).Round(time.Second))
	}
	if waiting > 0 {
		s += fmt.Sprintf(", %d message(s) waiting", waiting)
	}
	return s
}

func statsText(l *store.LastReply) string {
	if l == nil {
		return "No replies yet."
	}
	s := l.Stats
	var sb strings.Builder
	fmt.Fprintf(&sb, "Last reply: %s, %s ago\n", l.Model, time.Since(l.At).Round(time.Second))
	fmt.Fprintf(&sb, "Output: %d tokens at %.1f tok/s (%s)\n", s.EvalCount, s.TokensPerSecond(), round(s.EvalDuration))
	fmt.Fprintf(&sb, "Prompt: %d tokens at %.0f tok/s (%s)\n", s.PromptEvalCount, s.PromptTokensPerSecond(), round(s.PromptEvalDuration))
	fmt.Fprintf(&sb, "Model load: %s\n", round(s.LoadDuration))
	fmt.Fprintf(&sb, "Total: %s\n", round(s.TotalDuration))
	sb.WriteString("(Output includes thinking tokens.)")
	return sb.String()
}

func round(d time.Duration) time.Duration { return d.Round(100 * time.Millisecond) }

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

// mask hides all but the last 4 characters of a phone number in logs.
func mask(s string) string {
	if len(s) <= 4 {
		return s
	}
	return "…" + s[len(s)-4:]
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
