package config

import "testing"

func TestLoad(t *testing.T) {
	t.Setenv("SIGNAL_NUMBER", "+15550000009")
	t.Setenv("ALLOWED_SENDERS", " +15550000001 , +15550000002,")
	t.Setenv("OLLAMA_URL", "http://10.0.0.5:11434/")
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Allowed) != 2 || !c.Allowed["+15550000002"] {
		t.Fatalf("allowed: %v", c.Allowed)
	}
	if c.OllamaURL != "http://10.0.0.5:11434" || c.HistoryLimit != 40 {
		t.Fatalf("%+v", c)
	}

	t.Setenv("ALLOWED_SENDERS", "")
	if _, err := Load(); err == nil {
		t.Fatal("expected error without ALLOWED_SENDERS")
	}
}
