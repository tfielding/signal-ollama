package format

import (
	"strings"
	"testing"
)

func TestToSignal(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"heading", "### 1.  What TCP Is  ", "**1.  What TCP Is**"},
		{"heading with bold", "## **Big** idea", "**Big idea**"},
		{"rule dropped", "a\n\n---\n\nb", "a\n\nb"},
		{"bullets", "- one  \n  - two\n* three", "• one\n  ◦ two\n• three"},
		{"numbered kept", "1. **First**", "1. **First**"},
		{"bold italic strike", "**b** *i* _u_ __bb__ ~~s~~ ***bi***", "**b** *i* *u* **bb** ~s~ **bi**"},
		{"stray markers escaped", "5 * 3, ~2s, a || b", `5 \* 3, \~2s, a \|\| b`},
		{"snake_case untouched", "use snake_case_name here", "use snake_case_name here"},
		{"inline code escaped", "run `a*b ~c`", "run `a\\*b \\~c`"},
		{"link", "see [RFC 793](https://x.org/793) or [https://a.b](https://a.b)", "see RFC 793 (https://x.org/793) or https://a.b"},
		{"code block", "```go\nx := a * b\n\n```", "`x := a \\* b`"},
		{"quote", "> hi **there**", "▍hi **there**"},
		{"blank lines collapsed", "a\n\n\n\nb\n\n", "a\n\nb"},
		{
			"table",
			"| Feature | Purpose |\n|---|:---:|\n| **Seq** | Order `x` |\n| Ack | Confirm |",
			"**Seq**\n**Purpose:** Order `x`\n\n**Ack**\n**Purpose:** Confirm",
		},
		{"pipe without table", "a | b", "a | b"},
	}
	for _, c := range cases {
		if got := ToSignal(c.in); got != c.want {
			t.Errorf("%s:\n got: %q\nwant: %q", c.name, got, c.want)
		}
	}
}

func TestNoUnescapedMarkersLeak(t *testing.T) {
	// Unbalanced emphasis must not leave a bare marker that would style the rest of the message.
	got := ToSignal("this **never closes and *this either")
	for i, r := range got {
		if (r == '*' || r == '~' || r == '`') && (i == 0 || got[i-1] != '\\') {
			t.Fatalf("bare %q at %d in %q", r, i, got)
		}
	}
	if !strings.Contains(got, `\*\*never`) {
		t.Fatalf("got %q", got)
	}
}
