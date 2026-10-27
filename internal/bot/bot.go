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
	"signal-ollama/internal/toolbox"
)

// Messenger is the Signal side (implemented by signal.Client).
type Messenger interface {
	Send(ctx context.Context, recipient, text string, styled bool, attachments []signal.Attachment) error
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
}

// maxToolRounds limits how many times the model can call tools before answering.
const maxToolRounds = 6

// maxToolFailures stops tool use early when calls keep failing.
const maxToolFailures = 3

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

	ctx     context.Context
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

// response is what gets sent back to the user.
type response struct {
	text      string
	fromModel bool // Markdown from the model (converted to Signal formatting)
	images    []toolbox.Image
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
	r := b.respond(sender, text)
	if r.text == "" && len(r.images) == 0 {
		return
	}
	if err := b.send(sender, r); err != nil {
		slog.Error("send failed", "to", mask(sender), "err", err)
		return
	}
	slog.Info("reply sent", "to", mask(sender), "chars", len(r.text), "images", len(r.images), "took", time.Since(start).Round(100*time.Millisecond))
}

// send delivers a reply. Model replies are Markdown, converted to Signal formatting;
// if the styled send fails, the raw text is sent as a fallback.
func (b *Bot) send(to string, r response) error {
	var att []signal.Attachment
	for i, img := range r.images {
		att = append(att, signal.Attachment{MIME: img.MIME, Name: fmt.Sprintf("image-%d%s", i+1, ext(img.MIME)), Data: img.Data})
	}
	if !r.fromModel {
		return b.msg.Send(b.ctx, to, r.text, false, att)
	}
	if err := b.msg.Send(b.ctx, to, format.ToSignal(r.text), true, att); err != nil {
		slog.Warn("styled send failed, retrying as plain text", "err", err)
		return b.msg.Send(b.ctx, to, r.text, false, att)
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

func instant(text string) bool {
	f := strings.Fields(strings.ToLower(text))
	return len(f) > 0 && (f[0] == "/status" || f[0] == "/stats" || f[0] == "/stop")
}

// respond produces the reply for one message.
func (b *Bot) respond(sender, text string) response {
	chat, err := b.store.Load(sender)
	if err != nil {
		slog.Error("load history", "err", err)
		return response{text: "Couldn't load chat history: " + err.Error()}
	}
	if strings.HasPrefix(text, "/") {
		return response{text: b.command(sender, &chat, text)}
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
		return response{text: offlineText}
	}
	info, _ := b.llm.Info(b.ctx, model)
	think := b.thinkValue(model, chat.Think)
	var tools []toolbox.Tool
	if b.opt.Tools != nil && info.Has("tools") {
		tools = b.opt.Tools.Tools(b.ctx)
	}

	if len(tools) > 0 {
		// Inserted after the main system prompt so it applies even with a custom SYSTEM_PROMPT.
		guide := ollama.Message{Role: "system", Content: toolGuidance}
		at := 0
		if b.opt.SystemPrompt != "" {
			at = 1
		}
		msgs = append(msgs[:at], append([]ollama.Message{guide}, msgs[at:]...)...)
	}

	slog.Info("asking model", "model", model, "history", len(chat.History), "think", chat.Think, "tools", len(tools))
	start := time.Now()
	b.mu.Lock()
	callCtx, cancel := context.WithCancel(b.ctx)
	defer cancel()
	b.active[sender] = activity{model: model, started: start, cancel: cancel}
	b.mu.Unlock()
	stopTyping := b.typing(sender)
	answer, images, stats, err := b.run(callCtx, sender, model, msgs, think, tools, info.Has("vision"))
	stopTyping()
	b.mu.Lock()
	delete(b.active, sender)
	b.mu.Unlock()
	if err != nil && callCtx.Err() != nil && b.ctx.Err() == nil {
		// Cancelled by /stop, which already replied. Nothing is saved.
		slog.Info("model call stopped", "model", model, "took", time.Since(start).Round(100*time.Millisecond))
		return response{}
	}
	if err != nil {
		slog.Error("ollama chat", "model", model, "took", time.Since(start).Round(100*time.Millisecond), "err", err)
		if unreachable(err) {
			return response{text: offlineText}
		}
		return response{text: "Model error: " + err.Error()}
	}
	slog.Info("model replied", "model", model, "took", time.Since(start).Round(100*time.Millisecond),
		"tokens", stats.EvalCount, "tok_per_s", fmt.Sprintf("%.1f", stats.TokensPerSecond()), "images", len(images))
	if answer == "" && len(images) == 0 {
		answer = "(empty reply)"
	}

	chat.Last = &store.LastReply{Model: model, At: time.Now(), Stats: stats}
	saved := answer
	if len(images) > 0 {
		saved += fmt.Sprintf("\n\n[%d image(s) were sent to the user with this reply.]", len(images))
	}
	chat.History = trim(append(chat.History,
		ollama.Message{Role: "user", Content: text},
		ollama.Message{Role: "assistant", Content: strings.TrimSpace(saved)}), b.opt.HistoryLimit)
	if err := b.store.Save(sender, chat); err != nil {
		slog.Error("save history", "err", err)
	}
	return response{text: answer, fromModel: true, images: images}
}

// run asks the model, executing any tool calls it makes, until it gives a final answer.
// Images returned by tools are collected to send to the user, and shown to the model
// too if it supports vision.
func (b *Bot) run(ctx context.Context, sender, model string, msgs []ollama.Message, think any,
	tools []toolbox.Tool, vision bool) (string, []toolbox.Image, ollama.Stats, error) {

	defs := make([]ollama.Tool, len(tools))
	for i, t := range tools {
		defs[i] = ollama.Tool{Type: "function", Function: ollama.ToolFunction{Name: t.Name, Description: t.Description, Parameters: t.Schema}}
	}
	var images []toolbox.Image
	var stats ollama.Stats
	failures := 0
	for round := 0; ; round++ {
		if round == maxToolRounds || failures >= maxToolFailures {
			if len(defs) > 0 {
				slog.Warn("no more tool calls for this message, asking for a final answer", "rounds", round, "failures", failures)
				msgs = append(msgs, ollama.Message{Role: "system", Content: "Tool use is over for this message. " +
					"Answer now with what you have. If the tools failed, say so plainly; do not guess values."})
			}
			defs = nil // force a final answer
		}
		res, err := b.llm.Chat(ctx, model, msgs, think, defs)
		if err != nil {
			return "", nil, stats, err
		}
		stats.Add(res.Stats)
		if len(res.ToolCalls) == 0 || len(defs) == 0 {
			if strings.TrimSpace(res.Content) == "" && len(images) == 0 {
				return emptyAnswer(failures, round), nil, stats, nil
			}
			return res.Content, images, stats, nil
		}

		msgs = append(msgs, ollama.Message{Role: "assistant", Content: res.Content, Thinking: res.Thinking, ToolCalls: res.ToolCalls})
		for _, tc := range res.ToolCalls {
			name := tc.Function.Name
			b.setTool(sender, name)
			start := time.Now()
			out, err := b.opt.Tools.Call(ctx, name, tc.Function.Arguments)
			b.setTool(sender, "")
			if ctx.Err() != nil {
				return "", nil, stats, ctx.Err()
			}
			attrs := []any{"tool", name, "args", argKeys(tc.Function.Arguments), "took", time.Since(start).Round(time.Millisecond),
				"images", len(out.Images), "is_error", out.IsError}
			attrs = append(attrs, "chars", len(out.Text))
			if len(out.Dropped) > 0 {
				attrs = append(attrs, "dropped_args", out.Dropped)
			}
			if out.Truncated > 0 {
				attrs = append(attrs, "truncated_chars", out.Truncated)
			}
			if err != nil {
				attrs = append(attrs, "err", err)
			} else if out.IsError {
				attrs = append(attrs, "error_text", snippet(out.Text, 200))
			}
			slog.Info("tool call", attrs...)
			if err != nil || out.IsError {
				failures++
			}

			msg := ollama.Message{Role: "tool", ToolName: name, Content: out.Text}
			if err != nil {
				msg.Content = "Tool failed: " + err.Error()
			} else if out.IsError {
				msg.Content = "Tool returned an error: " + out.Text
			}
			for _, img := range out.Images {
				images = append(images, img)
				if vision {
					msg.Images = append(msg.Images, img.Data)
				}
			}
			if len(out.Images) > 0 {
				note := "[%d image(s) returned. They will be sent to the user with your reply; you cannot see them.]"
				if vision {
					note = "[%d image(s) attached. They will also be sent to the user with your reply.]"
				}
				msg.Content = strings.TrimSpace(msg.Content + "\n" + fmt.Sprintf(note, len(out.Images)))
			}
			msgs = append(msgs, msg)
		}
	}
}

// emptyAnswer explains a blank reply instead of sending nothing useful.
func emptyAnswer(failures, rounds int) string {
	if failures > 0 {
		return fmt.Sprintf("Sorry, I couldn't get an answer: %d tool call(s) failed. Try rephrasing, or check the bot's logs for the tool error.", failures)
	}
	if rounds > 0 {
		return "Sorry, the model returned no answer after using its tools. Try asking again."
	}
	return "Sorry, the model returned an empty answer. Try asking again, or /reset if this keeps happening."
}

func snippet(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > n {
		return s[:n] + "…"
	}
	return s
}

func (b *Bot) setTool(sender, tool string) {
	b.mu.Lock()
	if a, ok := b.active[sender]; ok {
		a.tool = tool
		b.active[sender] = a
	}
	b.mu.Unlock()
}

// argKeys logs argument names only, not values.
func argKeys(args map[string]any) []string {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	return keys
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
		} else if info, err := b.thinkInfo(b.ctx, current); err == nil {
			thinking = thinkLabel(chat.Think, info)
		}
		return fmt.Sprintf("Model: %s\nThinking: %s\nMemory: %d messages (max %d)\nOllama: %s\n%sNow: %s",
			current, thinking, len(chat.History), b.opt.HistoryLimit, ollamaState, b.toolsStatus(), b.now(sender))
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
			if info, err := b.thinkInfo(b.ctx, name); err == nil && info.Known && !info.Supports(chat.Think) {
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

// toolGuidance is added to the prompt whenever tools are available. It is deliberately
// generic: nothing here knows what the tools are.
const toolGuidance = "You have tools that provide live information and actions. " +
	"Use them whenever an answer depends on current, real-world state instead of guessing or saying you can't. " +
	"That state changes: call the tools again rather than relying on earlier messages in this conversation. " +
	"Pass only the parameters a tool defines, and leave optional filters out unless you're sure of the exact value. " +
	"If a tool returns an error or finds nothing, read the message and try again with fewer filters or none; do not repeat the same call. " +
	"Never invent readings or devices: report only what the tools returned, and say so if they failed. " +
	"Only offer to do things you have a tool for."

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

func (b *Bot) thinkInfo(ctx context.Context, model string) (ollama.ThinkInfo, error) {
	info, err := b.llm.Info(ctx, model)
	return info.Think, err
}

// thinkValue returns the API value for a chat's thinking setting, or nil (model default)
// if it's unset or the model doesn't support it.
func (b *Bot) thinkValue(model, setting string) any {
	if setting == "" {
		return nil
	}
	if info, err := b.thinkInfo(b.ctx, model); err == nil && info.Known && !info.Supports(setting) {
		slog.Info("thinking setting not supported by model, using default", "model", model, "think", setting)
		return nil
	}
	return ollama.ThinkValue(setting)
}

func (b *Bot) thinkCommand(sender string, chat *store.Chat, model string, args []string) string {
	info, err := b.thinkInfo(b.ctx, model)
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

// toolsStatus is the "Tools:" line for /status (empty if no MCP servers are configured).
func (b *Bot) toolsStatus() string {
	if b.opt.Tools == nil {
		return ""
	}
	tools := b.opt.Tools.Tools(b.ctx)
	names := make([]string, len(tools))
	for i, t := range tools {
		names[i] = t.Name
	}
	line := "Tools: none"
	if len(names) > 0 {
		line = "Tools: " + strings.Join(names, ", ")
	}
	for server, err := range b.opt.Tools.Errors() {
		line += fmt.Sprintf("\n%s: unavailable (%s)", server, short(err))
	}
	return line + "\n"
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
		if a.tool != "" {
			s += " (running " + a.tool + ")"
		}
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
