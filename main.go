// Command signal-ollama connects a Signal number (via signal-cli-rest-api) to an Ollama server.
package main

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"signal-ollama/internal/bot"
	"signal-ollama/internal/config"
	"signal-ollama/internal/ollama"
	sig "signal-ollama/internal/signal"
	"signal-ollama/internal/store"
	"signal-ollama/internal/toolbox"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	llm := ollama.New(cfg.OllamaURL, cfg.OllamaTimeout)
	if cfg.DefaultModel == "" {
		cfg.DefaultModel, err = firstModel(ctx, llm)
		if err != nil {
			return err
		}
	}

	st, err := store.New(cfg.DataDir)
	if err != nil {
		return err
	}
	sc := sig.New(cfg.SignalURL, cfg.SignalNumber)
	var tools bot.Tools
	if len(cfg.MCPServers) > 0 {
		box := toolbox.New(cfg.MCPServers, toolbox.Options{Allowed: cfg.MCPTools, AllowWrite: cfg.MCPAllowWrite})
		defer box.Close()
		names := []string{}
		for _, t := range box.Tools(ctx) {
			names = append(names, t.Name)
		}
		slog.Info("mcp tools", "servers", len(cfg.MCPServers), "tools", names)
		tools = box
	}
	b := bot.New(ctx, sc, llm, st, bot.Options{
		Tools:        tools,
		OwnNumber:    cfg.SignalNumber,
		Allowed:      cfg.Allowed,
		DefaultModel: cfg.DefaultModel,
		SystemPrompt: cfg.SystemPrompt,
		HistoryLimit: cfg.HistoryLimit,
	})

	slog.Info("starting", "signal", cfg.SignalURL, "ollama", cfg.OllamaURL, "model", cfg.DefaultModel, "allowed", len(cfg.Allowed))
	err = sc.Listen(ctx, b.Handle)
	b.Wait()
	if errors.Is(err, context.Canceled) {
		slog.Info("shut down")
		return nil
	}
	return err
}

// firstModel waits for Ollama to be reachable and returns its first installed model.
func firstModel(ctx context.Context, llm *ollama.Client) (string, error) {
	for {
		models, err := llm.Models(ctx)
		if err == nil && len(models) > 0 {
			slog.Info("OLLAMA_MODEL not set, using first installed model", "model", models[0])
			return models[0], nil
		}
		if err == nil {
			err = errors.New("no models installed (run: ollama pull <model>)")
		}
		slog.Warn("waiting for ollama", "err", err)
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(10 * time.Second):
		}
	}
}
