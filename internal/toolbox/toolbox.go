// Package toolbox connects to MCP servers and exposes their tools to the model.
// It knows nothing about any particular tool: servers are configured by URL.
package toolbox

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	refreshEvery  = time.Minute
	listTimeout   = 5 * time.Second // don't hold up a reply for a server that's down
	callTimeout   = 60 * time.Second
	maxResultText = 8000
)

// Server is one MCP server (Streamable HTTP).
type Server struct {
	Name  string
	URL   string
	Token string // optional bearer token
}

// Tool is a tool as exposed to the model.
type Tool struct {
	Name        string // name the model sees (prefixed with the server name only on collisions)
	Description string
	Schema      any
	ReadOnly    bool
	server      string
	remote      string // name on the server
}

// Image is image content returned by a tool.
type Image struct {
	MIME string
	Data []byte
}

// Output is a tool result, reduced to what the bot needs.
type Output struct {
	Text    string
	Images  []Image
	IsError bool
	Dropped []string // arguments removed because the tool doesn't accept them
}

type Options struct {
	Allowed    map[string]bool // if non-empty, only these tool names are exposed
	AllowWrite bool            // expose tools that aren't marked read-only
}

type Box struct {
	servers []Server
	opt     Options
	client  *mcp.Client

	mu       sync.Mutex
	sessions map[string]*mcp.ClientSession
	tools    []Tool
	errs     map[string]error // last listing error per server
	fetched  time.Time
	hidden   map[string]bool // tools already logged as hidden
}

func New(servers []Server, opt Options) *Box {
	return &Box{
		servers:  servers,
		opt:      opt,
		client:   mcp.NewClient(&mcp.Implementation{Name: "signal-ollama", Version: "1"}, nil),
		sessions: map[string]*mcp.ClientSession{},
		errs:     map[string]error{},
		hidden:   map[string]bool{},
	}
}

// Tools returns the exposed tools, refreshing the list at most once a minute.
// Servers that are down are skipped (see Errors).
func (b *Box) Tools(ctx context.Context) []Tool {
	b.mu.Lock()
	fresh := time.Since(b.fetched) < refreshEvery
	tools := b.tools
	b.mu.Unlock()
	if fresh {
		return tools
	}

	var all []Tool
	errs := map[string]error{}
	for _, s := range b.servers {
		lctx, cancel := context.WithTimeout(ctx, listTimeout)
		ts, err := b.list(lctx, s)
		cancel()
		if err != nil {
			slog.Warn("mcp server unavailable", "server", s.Name, "err", err)
			errs[s.Name] = err
			continue
		}
		all = append(all, ts...)
	}
	all = b.filter(dedupe(all))

	b.mu.Lock()
	b.tools, b.errs, b.fetched = all, errs, time.Now()
	b.mu.Unlock()
	return all
}

// Errors returns the servers whose last listing failed.
func (b *Box) Errors() map[string]error {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := map[string]error{}
	for k, v := range b.errs {
		out[k] = v
	}
	return out
}

// Call runs a tool by its exposed name.
func (b *Box) Call(ctx context.Context, name string, args map[string]any) (Output, error) {
	var tool *Tool
	for _, t := range b.Tools(ctx) {
		if t.Name == name {
			tool = &t
			break
		}
	}
	if tool == nil {
		return Output{}, fmt.Errorf("no tool named %q is available", name)
	}
	srv, _ := b.server(tool.server)
	ctx, cancel := context.WithTimeout(ctx, callTimeout)
	defer cancel()
	args, dropped := cleanArgs(tool.Schema, args)

	var res *mcp.CallToolResult
	err := b.withSession(ctx, srv, func(cs *mcp.ClientSession) error {
		var err error
		res, err = cs.CallTool(ctx, &mcp.CallToolParams{Name: tool.remote, Arguments: args})
		return err
	})
	if err != nil {
		return Output{Dropped: dropped}, err
	}
	out := convert(res)
	out.Dropped = dropped
	return out, nil
}

// cleanArgs removes arguments the tool's input schema doesn't declare. Small models
// often invent parameters (e.g. passing "area" to a tool that takes none), which
// makes the server reject the call. Tools that accept extra properties are left alone.
func cleanArgs(schema any, args map[string]any) (map[string]any, []string) {
	clean := map[string]any{}
	if len(args) == 0 {
		return clean, nil
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return args, nil
	}
	var s struct {
		Properties           map[string]json.RawMessage `json:"properties"`
		AdditionalProperties json.RawMessage            `json:"additionalProperties"`
	}
	if json.Unmarshal(raw, &s) != nil {
		return args, nil
	}
	extraOK := len(s.AdditionalProperties) > 0 && string(s.AdditionalProperties) != "false"
	if extraOK || (s.Properties == nil && len(s.AdditionalProperties) == 0 && !emptyObject(raw)) {
		return args, nil // schema allows anything, or says nothing we can rely on
	}
	var dropped []string
	for k, v := range args {
		if _, ok := s.Properties[k]; ok {
			clean[k] = v
		} else {
			dropped = append(dropped, k)
		}
	}
	sort.Strings(dropped)
	return clean, dropped
}

// emptyObject reports whether a schema is {"type":"object"} with no properties at all,
// which for tool inputs means "takes no arguments".
func emptyObject(raw []byte) bool {
	var s map[string]any
	if json.Unmarshal(raw, &s) != nil {
		return false
	}
	_, hasProps := s["properties"]
	return s["type"] == "object" && !hasProps
}

func convert(res *mcp.CallToolResult) Output {
	out := Output{IsError: res.IsError}
	var text []string
	for _, c := range res.Content {
		switch v := c.(type) {
		case *mcp.TextContent:
			text = append(text, v.Text)
		case *mcp.ImageContent:
			out.Images = append(out.Images, Image{MIME: v.MIMEType, Data: v.Data})
		}
	}
	out.Text = strings.Join(text, "\n")
	if len(out.Text) > maxResultText {
		out.Text = out.Text[:maxResultText] + "\n…(truncated)"
	}
	return out
}

func (b *Box) list(ctx context.Context, s Server) ([]Tool, error) {
	var tools []Tool
	err := b.withSession(ctx, s, func(cs *mcp.ClientSession) error {
		tools = nil
		for t, err := range cs.Tools(ctx, nil) {
			if err != nil {
				return err
			}
			ro := t.Annotations != nil && t.Annotations.ReadOnlyHint
			tools = append(tools, Tool{
				Name: t.Name, Description: t.Description, Schema: t.InputSchema,
				ReadOnly: ro, server: s.Name, remote: t.Name,
			})
		}
		return nil
	})
	return tools, err
}

// withSession runs fn on a session to s, reconnecting once if the session went bad
// (e.g. the server restarted).
func (b *Box) withSession(ctx context.Context, s Server, fn func(*mcp.ClientSession) error) error {
	for attempt := 0; ; attempt++ {
		cs, err := b.session(ctx, s)
		if err != nil {
			return err
		}
		err = fn(cs)
		if err == nil || attempt > 0 || ctx.Err() != nil {
			return err
		}
		slog.Info("mcp call failed, reconnecting", "server", s.Name, "err", err)
		b.drop(s.Name, cs)
	}
}

func (b *Box) session(ctx context.Context, s Server) (*mcp.ClientSession, error) {
	b.mu.Lock()
	cs := b.sessions[s.Name]
	b.mu.Unlock()
	if cs != nil {
		return cs, nil
	}
	hc := &http.Client{Timeout: callTimeout + 5*time.Second}
	if s.Token != "" {
		hc.Transport = bearer{s.Token}
	}
	cs, err := b.client.Connect(ctx, &mcp.StreamableClientTransport{
		Endpoint: s.URL, HTTPClient: hc, DisableStandaloneSSE: true, MaxRetries: -1,
	}, nil)
	if err != nil {
		return nil, err
	}
	b.mu.Lock()
	if existing := b.sessions[s.Name]; existing != nil { // lost a race; keep the first
		b.mu.Unlock()
		cs.Close()
		return existing, nil
	}
	b.sessions[s.Name] = cs
	b.mu.Unlock()
	slog.Info("connected to mcp server", "server", s.Name, "url", s.URL)
	return cs, nil
}

func (b *Box) drop(name string, cs *mcp.ClientSession) {
	b.mu.Lock()
	if b.sessions[name] == cs {
		delete(b.sessions, name)
	}
	b.mu.Unlock()
	cs.Close()
}

func (b *Box) server(name string) (Server, bool) {
	for _, s := range b.servers {
		if s.Name == name {
			return s, true
		}
	}
	return Server{}, false
}

// filter applies the allowlist and the read-only rule.
func (b *Box) filter(tools []Tool) []Tool {
	var out []Tool
	for _, t := range tools {
		if len(b.opt.Allowed) > 0 && !b.opt.Allowed[t.Name] {
			continue
		}
		if !t.ReadOnly && !b.opt.AllowWrite {
			b.mu.Lock()
			first := !b.hidden[t.server+"/"+t.Name]
			b.hidden[t.server+"/"+t.Name] = true
			b.mu.Unlock()
			if first {
				slog.Info("hiding tool not marked read-only (allow with MCP_TOOLS + MCP_ALLOW_WRITE=true)", "tool", t.Name, "server", t.server)
			}
			continue
		}
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// dedupe prefixes tool names with their server name when two servers use the same name.
func dedupe(tools []Tool) []Tool {
	count := map[string]int{}
	for _, t := range tools {
		count[t.Name]++
	}
	for i := range tools {
		if count[tools[i].Name] > 1 {
			tools[i].Name = tools[i].server + "_" + tools[i].Name
		}
	}
	return tools
}

// Close ends all sessions.
func (b *Box) Close() {
	b.mu.Lock()
	defer b.mu.Unlock()
	for name, cs := range b.sessions {
		cs.Close()
		delete(b.sessions, name)
	}
}

// ParseServers parses "name=url,name2=url2". tokens maps server name to bearer token.
func ParseServers(spec string, token func(name string) string) ([]Server, error) {
	var out []Server
	seen := map[string]bool{}
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, url, ok := strings.Cut(part, "=")
		name, url = strings.TrimSpace(name), strings.TrimSpace(url)
		if !ok || name == "" || !(strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://")) {
			return nil, fmt.Errorf("MCP_SERVERS entry %q: want name=http://host:port/mcp", part)
		}
		if seen[name] {
			return nil, fmt.Errorf("MCP_SERVERS: duplicate name %q", name)
		}
		seen[name] = true
		out = append(out, Server{Name: name, URL: url, Token: token(name)})
	}
	return out, nil
}

type bearer struct{ token string }

func (t bearer) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+t.token)
	return http.DefaultTransport.RoundTrip(r)
}
