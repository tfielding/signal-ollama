package bot

// Slash commands (/status, /model, /stop, ...).

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/tfielding/signal-ollama/internal/store"
)

func instant(text string) bool {
	f := strings.Fields(strings.ToLower(text))
	return len(f) > 0 && (f[0] == "/status" || f[0] == "/stats" || f[0] == "/stop")
}

func (b *Bot) command(ctx context.Context, sender string, chat *store.Chat, text string) string {
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
		if err := b.llm.Ping(ctx); err != nil {
			ollamaState = "offline (" + short(err) + ")"
		} else if info, err := b.thinkInfo(ctx, current); err == nil {
			thinking = thinkLabel(chat.Think, info)
		}
		return fmt.Sprintf("Model: %s\nThinking: %s\nMemory: %d messages (max %d)\nOllama: %s\n%sNow: %s",
			current, thinking, len(chat.History), b.opt.HistoryLimit, ollamaState, b.toolsStatus(ctx), b.now(sender))
	case "/think":
		return b.thinkCommand(ctx, sender, chat, current, args)
	case "/stop":
		return b.stop(sender)
	case "/stats":
		return statsText(chat.Last)
	case "/model":
		models, err := b.llm.Models(ctx)
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
			if info, err := b.thinkInfo(ctx, name); err == nil && info.Known && !info.Supports(chat.Think) {
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

// toolsStatus is the "Tools:" line for /status (empty if no MCP servers are configured).
func (b *Bot) toolsStatus(ctx context.Context) string {
	if b.opt.Tools == nil {
		return ""
	}
	tools := b.opt.Tools.Tools(ctx)
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
