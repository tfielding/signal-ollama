// Package store persists per-chat state as JSON files.
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"time"

	"github.com/tfielding/signal-ollama/internal/ollama"
)

type Chat struct {
	Model   string           `json:"model,omitempty"` // empty = default model
	Think   string           `json:"think,omitempty"` // "" (model default), on, off, low, medium, high
	History []ollama.Message `json:"history"`
	Last    *LastReply       `json:"last,omitempty"` // timings of the most recent reply, for /stats
}

type LastReply struct {
	Model string       `json:"model"`
	At    time.Time    `json:"at"`
	Stats ollama.Stats `json:"stats"`
}

type Store struct{ dir string }

func New(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, err
	}
	return &Store{dir: dir}, nil
}

// Load returns the chat for id, or an empty chat if none is saved.
func (s *Store) Load(id string) (Chat, error) {
	var c Chat
	b, err := os.ReadFile(s.path(id))
	if errors.Is(err, os.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return c, err
	}
	return c, json.Unmarshal(b, &c)
}

// Save writes the chat atomically.
func (s *Store) Save(id string, c Chat) error {
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path(id) + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.path(id))
}

// File names are hashed so phone numbers don't appear on disk as names.
func (s *Store) path(id string) string {
	h := sha256.Sum256([]byte(id))
	return filepath.Join(s.dir, hex.EncodeToString(h[:8])+".json")
}
