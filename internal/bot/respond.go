package bot

// Answering a chat message: the model call and the tool-use loop.

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strings"
	"time"

	"github.com/tfielding/signal-ollama/internal/ollama"
	"github.com/tfielding/signal-ollama/internal/store"
	"github.com/tfielding/signal-ollama/internal/toolbox"
)

// maxToolRounds limits how many times the model can call tools before answering.
const maxToolRounds = 6

// maxToolFailures stops tool use early when calls keep failing.
const maxToolFailures = 3

// response is what gets sent back to the user.
type response struct {
	text      string
	fromModel bool // Markdown from the model (converted to Signal formatting)
	images    []toolbox.Image
}

// respond produces the reply for one message.
func (b *Bot) respond(ctx context.Context, sender, text string) response {
	chat, err := b.store.Load(sender)
	if err != nil {
		slog.Error("load history", "err", err)
		return response{text: "Couldn't load chat history: " + err.Error()}
	}
	if strings.HasPrefix(text, "/") {
		return response{text: b.command(ctx, sender, &chat, text)}
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
	if err := b.llm.Ping(ctx); err != nil {
		slog.Warn("ollama unreachable", "err", err)
		return response{text: offlineText}
	}
	info, _ := b.llm.Info(ctx, model)
	think := b.thinkValue(ctx, model, chat.Think)
	var tools []toolbox.Tool
	if b.opt.Tools != nil && info.Has("tools") {
		tools = b.opt.Tools.Tools(ctx)
	}

	if len(tools) > 0 {
		// Inserted after the main system prompt so it applies even with a custom SYSTEM_PROMPT.
		content := toolGuidance
		if extra := b.opt.Tools.Guidance(tools); extra != "" {
			content += "\n\n" + extra
		}
		guide := ollama.Message{Role: "system", Content: content}
		at := 0
		if b.opt.SystemPrompt != "" {
			at = 1
		}
		msgs = append(msgs[:at], append([]ollama.Message{guide}, msgs[at:]...)...)
	}

	slog.Info("asking model", "model", model, "history", len(chat.History), "think", chat.Think, "tools", len(tools))
	start := time.Now()
	b.mu.Lock()
	callCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	b.active[sender] = activity{model: model, started: start, cancel: cancel}
	b.mu.Unlock()
	stopTyping := b.typing(ctx, sender)
	answer, images, stats, err := b.run(callCtx, sender, model, msgs, think, tools, info.Has("vision"))
	stopTyping()
	b.mu.Lock()
	delete(b.active, sender)
	b.mu.Unlock()
	if err != nil && callCtx.Err() != nil && ctx.Err() == nil {
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
			if out.Retried != "" {
				attrs = append(attrs, "retried_without_args", snippet(out.Retried, 120))
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
func (b *Bot) typing(ctx context.Context, to string) (stop func()) {
	done := make(chan struct{})
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		t := time.NewTicker(b.opt.TypingRefresh)
		defer t.Stop()
		for {
			if err := b.msg.Typing(ctx, to, true); err != nil {
				slog.Warn("typing indicator failed", "err", err)
			}
			select {
			case <-done:
				// The call may have been cancelled (/stop), so use a fresh context.
				offCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
				_ = b.msg.Typing(offCtx, to, false)
				cancel()
				return
			case <-ctx.Done():
				return
			case <-t.C:
			}
		}
	}()
	return func() { close(done); <-finished }
}

// toolGuidance is added to the prompt whenever tools are available. It is deliberately
// generic: nothing here knows what the tools are.
const toolGuidance = "You have tools that can look things up or take actions. " +
	"Use them whenever an answer depends on information they can provide, instead of guessing or saying you can't. " +
	"Information from tools can change: call them again rather than relying on earlier messages in this conversation. " +
	"Pass only the parameters a tool defines, and leave optional parameters out unless you're sure of the exact value. " +
	"If a tool returns an error or nothing useful, read the message and try a different call (often with fewer parameters); do not repeat the same call. " +
	"Never invent facts or data: report only what the tools returned, and say so if they failed. " +
	"Only offer to do things you have a tool for."

const offlineText = "Can't reach Ollama right now. The machine running it may be asleep or offline. Try again in a bit."

// unreachable reports whether err means Ollama couldn't be contacted (as opposed to a slow or failed reply).
func unreachable(err error) bool {
	var oe *net.OpError
	return errors.As(err, &oe) && !oe.Timeout()
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
