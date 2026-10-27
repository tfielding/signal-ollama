package toolbox

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func server(t *testing.T, name string, tools ...string) *httptest.Server {
	s := mcp.NewServer(&mcp.Implementation{Name: name, Version: "1"}, nil)
	for _, tn := range tools {
		tn := tn
		mcp.AddTool(s, &mcp.Tool{Name: tn, Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
			func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, any, error) {
				return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: name + ":" + tn + ":" + strings.Repeat("x", 9000)}}}, nil, nil
			})
	}
	srv := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return s }, nil))
	t.Cleanup(srv.Close)
	return srv
}

func names(ts []Tool) string {
	var n []string
	for _, t := range ts {
		n = append(n, t.Name)
	}
	return strings.Join(n, ",")
}

func TestCollisionsAllowlistAndDownServer(t *testing.T) {
	a := server(t, "a", "status", "snap")
	b := server(t, "b", "status")
	box := New([]Server{{Name: "a", URL: a.URL}, {Name: "b", URL: b.URL}, {Name: "down", URL: "http://127.0.0.1:1/mcp"}}, Options{})
	defer box.Close()
	ctx := context.Background()

	if got := names(box.Tools(ctx)); got != "a_status,b_status,snap" {
		t.Fatalf("tools: %s", got)
	}
	if _, ok := box.Errors()["down"]; !ok {
		t.Fatal("expected error for down server")
	}
	out, err := box.Call(ctx, "b_status", nil)
	if err != nil || !strings.HasPrefix(out.Text, "b:status:") || out.Truncated != 0 {
		t.Fatalf("call: %q %v", out.Text[:20], err)
	}
	if _, err := box.Call(ctx, "nope", nil); err == nil {
		t.Fatal("expected error for unknown tool")
	}

	only := New([]Server{{Name: "a", URL: a.URL}}, Options{Allowed: map[string]bool{"snap": true}})
	defer only.Close()
	if got := names(only.Tools(ctx)); got != "snap" {
		t.Fatalf("allowlist: %s", got)
	}
}

func TestReconnectsAfterServerRestart(t *testing.T) {
	var gen atomic.Int32
	mk := func() *mcp.Server {
		s := mcp.NewServer(&mcp.Implementation{Name: "r", Version: "1"}, nil)
		mcp.AddTool(s, &mcp.Tool{Name: "ping", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
			func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, any, error) {
				return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "pong"}}}, nil, nil
			})
		return s
	}
	// Each "generation" is a fresh handler with no memory of old sessions, like a restarted server.
	handlers := []http.Handler{
		mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return mk() }, nil),
		mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return mk() }, nil),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handlers[gen.Load()].ServeHTTP(w, r)
	}))
	defer srv.Close()

	box := New([]Server{{Name: "r", URL: srv.URL}}, Options{})
	defer box.Close()
	ctx := context.Background()
	if out, err := box.Call(ctx, "ping", nil); err != nil || out.Text != "pong" {
		t.Fatalf("first call: %v %v", out, err)
	}
	gen.Store(1) // "restart": old session id is now unknown
	if out, err := box.Call(ctx, "ping", nil); err != nil || out.Text != "pong" {
		t.Fatalf("after restart: %v %v", out, err)
	}
}

func TestParseServers(t *testing.T) {
	got, err := ParseServers(" go2rtc=http://go2rtc-mcp:8090/mcp , ha=https://ha/mcp", func(n string) string { return "t-" + n })
	if err != nil || len(got) != 2 || got[0].Name != "go2rtc" || got[0].URL != "http://go2rtc-mcp:8090/mcp" || got[1].Token != "t-ha" {
		t.Fatalf("%+v %v", got, err)
	}
	for _, bad := range []string{"noequals", "x=ftp://a", "a=http://x,a=http://y", "=http://x"} {
		if _, err := ParseServers(bad, func(string) string { return "" }); err == nil {
			t.Errorf("%q: expected error", bad)
		}
	}
	if got, err := ParseServers("", nil); err != nil || len(got) != 0 {
		t.Fatalf("empty: %v %v", got, err)
	}
}

func TestCleanArgs(t *testing.T) {
	none := map[string]any{"type": "object", "additionalProperties": false}
	noneNoAP := map[string]any{"type": "object"}
	some := map[string]any{"type": "object", "properties": map[string]any{"camera": map[string]any{"type": "string"}}, "additionalProperties": false}
	open := map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{}}, "additionalProperties": true}
	args := map[string]any{"camera": "x", "area": "Kitchen", "domain": "light"}

	cases := []struct {
		name    string
		schema  any
		keep    string
		dropped string
	}{
		{"no params (additionalProperties false)", none, "", "area,camera,domain"},
		{"no params (bare object)", noneNoAP, "", "area,camera,domain"},
		{"declared params", some, "camera", "area,domain"},
		{"additional allowed", open, "area,camera,domain", ""},
		{"no schema", nil, "area,camera,domain", ""},
	}
	for _, c := range cases {
		got, dropped := cleanArgs(c.schema, args)
		keys := []string{}
		for k := range got {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if strings.Join(keys, ",") != c.keep || strings.Join(dropped, ",") != c.dropped {
			t.Errorf("%s: kept %v dropped %v", c.name, keys, dropped)
		}
	}
}

func TestInventedArgsDroppedEndToEnd(t *testing.T) {
	s := mcp.NewServer(&mcp.Implementation{Name: "ha", Version: "1"}, nil)
	mcp.AddTool(s, &mcp.Tool{Name: "GetLiveContext", Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true}},
		func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "Kitchen light: on"}}}, nil, nil
		})
	srv := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return s }, nil))
	defer srv.Close()
	box := New([]Server{{Name: "ha", URL: srv.URL}}, Options{})
	defer box.Close()

	out, err := box.Call(context.Background(), "GetLiveContext", map[string]any{"area": "Kitchen", "domain": "light"})
	if err != nil || out.IsError || out.Text != "Kitchen light: on" || strings.Join(out.Dropped, ",") != "area,domain" {
		t.Fatalf("got %+v err=%v", out, err)
	}
}

func TestUnwrapAndTruncate(t *testing.T) {
	ha := `{"result": "Live Context:\n- names: Basement Door\n  domain: lock\n  state: locked"}`
	if got := unwrap(ha); got != "Live Context:\n- names: Basement Door\n  domain: lock\n  state: locked" {
		t.Fatalf("unwrap: %q", got)
	}
	for _, keep := range []string{`{"cameras":["a"]}`, `{"a":"x","b":"y"}`, `plain text`, `{"n": 5}`} {
		if unwrap(keep) != keep {
			t.Errorf("should be unchanged: %s", keep)
		}
	}

	long := strings.Repeat("é", 3000) // 2 bytes each: cut must not split a character
	out := convert(&mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: long}}}, 1001)
	if out.Truncated == 0 || !utf8.ValidString(out.Text) || !strings.Contains(out.Text, "more characters not shown") {
		t.Fatalf("truncate: %d valid=%v", out.Truncated, utf8.ValidString(out.Text))
	}
}
