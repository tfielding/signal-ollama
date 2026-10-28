package bot

// The /think setting: per-chat control of model reasoning.

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/tfielding/signal-ollama/internal/ollama"
	"github.com/tfielding/signal-ollama/internal/store"
)

func (b *Bot) thinkInfo(ctx context.Context, model string) (ollama.ThinkInfo, error) {
	info, err := b.llm.Info(ctx, model)
	return info.Think, err
}

// thinkValue returns the API value for a chat's thinking setting, or nil (model default)
// if it's unset or the model doesn't support it.
func (b *Bot) thinkValue(ctx context.Context, model, setting string) any {
	if setting == "" {
		return nil
	}
	if info, err := b.thinkInfo(ctx, model); err == nil && info.Known && !info.Supports(setting) {
		slog.Info("thinking setting not supported by model, using default", "model", model, "think", setting)
		return nil
	}
	return ollama.ThinkValue(setting)
}

func (b *Bot) thinkCommand(ctx context.Context, sender string, chat *store.Chat, model string, args []string) string {
	info, err := b.thinkInfo(ctx, model)
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
