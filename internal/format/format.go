// Package format converts Markdown from the model into Signal's styled text.
//
// signal-cli-rest-api (text_mode "styled") understands **bold**, *italic*,
// ~strikethrough~, `monospace` and ||spoiler||, with \ as an escape. Everything
// else Markdown uses (headings, tables, rules, links, code fences) is rewritten
// into something that reads well on a phone, and stray marker characters are
// escaped so they show up literally.
package format

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	fenceRe   = regexp.MustCompile("^\\s*(```|~~~)")
	headingRe = regexp.MustCompile(`^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$`)
	ruleRe    = regexp.MustCompile(`^\s{0,3}([-*_])(\s*[-*_]){2,}\s*$`)
	bulletRe  = regexp.MustCompile(`^(\s*)[-*+]\s+(.*)$`)
	quoteRe   = regexp.MustCompile(`^\s{0,3}>\s?(.*)$`)
	tableSep  = regexp.MustCompile(`^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$`)

	codeRe     = regexp.MustCompile("`([^`\n]+)`")
	imageRe    = regexp.MustCompile(`!\[([^\]]*)\]\(([^)\s]+)[^)]*\)`)
	linkRe     = regexp.MustCompile(`\[([^\]]+)\]\(([^)\s]+)[^)]*\)`)
	boldItalRe = regexp.MustCompile(`\*\*\*([^*\n]+?)\*\*\*`)
	boldRe     = regexp.MustCompile(`\*\*([^*\n]+?)\*\*`)
	boldUndRe  = regexp.MustCompile(`__([^_\n]+?)__`)
	italRe     = regexp.MustCompile(`(^|[^\w*\\])\*([^*\s](?:[^*\n]*[^*\s])?)\*`)
	italUndRe  = regexp.MustCompile(`(^|[^\w])_([^_\s](?:[^_\n]*[^_\s])?)_`)
	strikeRe   = regexp.MustCompile(`~~([^~\n]+?)~~`)
	holdRe     = regexp.MustCompile(`\x{E000}(\d+)\x{E001}`) // placeholder: private-use runes around an index
)

// ToSignal converts Markdown to Signal styled text.
func ToSignal(md string) string {
	lines := strings.Split(strings.ReplaceAll(md, "\r\n", "\n"), "\n")
	var out []string
	for i := 0; i < len(lines); i++ {
		line := lines[i]

		// Code block: monospace, one styled run per line.
		if fenceRe.MatchString(line) {
			for i++; i < len(lines) && !fenceRe.MatchString(lines[i]); i++ {
				if strings.TrimSpace(lines[i]) == "" {
					out = append(out, "")
				} else {
					out = append(out, "`"+escape(lines[i])+"`")
				}
			}
			continue
		}

		// Table: header row followed by a separator row.
		if strings.Contains(line, "|") && i+1 < len(lines) && tableSep.MatchString(lines[i+1]) {
			header := cells(line)
			i += 2
			for ; i < len(lines) && strings.Contains(lines[i], "|") && strings.TrimSpace(lines[i]) != ""; i++ {
				out = append(out, tableRow(header, cells(lines[i]))...)
				out = append(out, "")
			}
			i--
			continue
		}

		switch {
		case ruleRe.MatchString(line):
			out = append(out, "")
		case headingRe.MatchString(line):
			out = append(out, bold(headingRe.FindStringSubmatch(line)[1]))
		case bulletRe.MatchString(line):
			m := bulletRe.FindStringSubmatch(line)
			level := len(strings.ReplaceAll(m[1], "\t", "  ")) / 2
			mark := "• "
			if level > 0 {
				mark = "◦ "
			}
			out = append(out, strings.Repeat("  ", level)+mark+inline(strings.TrimRight(m[2], " \t")))
		case quoteRe.MatchString(line):
			out = append(out, "▍"+inline(quoteRe.FindStringSubmatch(line)[1]))
		default:
			out = append(out, inline(strings.TrimRight(line, " \t")))
		}
	}
	return collapse(out)
}

// tableRow renders one row as a bold title (first cell) followed by "Header: value" lines.
func tableRow(header, row []string) []string {
	var lines []string
	for j, c := range row {
		if c == "" {
			continue
		}
		if j == 0 {
			lines = append(lines, bold(c))
			continue
		}
		if j < len(header) && header[j] != "" {
			lines = append(lines, bold(header[j]+":")+" "+inline(c))
		} else {
			lines = append(lines, inline(c))
		}
	}
	return lines
}

func cells(line string) []string {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "|")
	line = strings.TrimSuffix(line, "|")
	parts := strings.Split(line, "|")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	return parts
}

// bold wraps text in bold, removing emphasis markers inside it so styles don't nest oddly.
func bold(s string) string {
	s = strings.NewReplacer("**", "", "__", "").Replace(s)
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	return "**" + inline(s) + "**"
}

// inline converts emphasis, code and links, and escapes any leftover marker characters.
func inline(s string) string {
	var held []string
	hold := func(x string) string {
		held = append(held, x)
		return fmt.Sprintf("\uE000%d\uE001", len(held)-1)
	}

	s = codeRe.ReplaceAllStringFunc(s, func(m string) string {
		return hold("`" + escape(m[1:len(m)-1]) + "`")
	})
	s = imageRe.ReplaceAllString(s, "$1 ($2)")
	s = linkRe.ReplaceAllStringFunc(s, func(m string) string {
		p := linkRe.FindStringSubmatch(m)
		if p[1] == p[2] {
			return p[2]
		}
		return p[1] + " (" + p[2] + ")"
	})
	wrap := func(re *regexp.Regexp, marker string) {
		s = re.ReplaceAllStringFunc(s, func(m string) string {
			return hold(marker + escape(re.FindStringSubmatch(m)[1]) + marker)
		})
	}
	wrap(boldItalRe, "**")
	wrap(boldRe, "**")
	wrap(boldUndRe, "**")
	wrap(strikeRe, "~")
	for _, re := range []*regexp.Regexp{italRe, italUndRe} {
		s = re.ReplaceAllStringFunc(s, func(m string) string {
			p := re.FindStringSubmatch(m)
			return p[1] + hold("*"+escape(p[2])+"*")
		})
	}

	s = escape(s)
	for holdRe.MatchString(s) { // held spans can contain other held spans
		s = holdRe.ReplaceAllStringFunc(s, func(m string) string {
			var n int
			fmt.Sscanf(holdRe.FindStringSubmatch(m)[1], "%d", &n)
			return held[n]
		})
	}
	return s
}

// escape makes marker characters literal for the styled-text parser.
func escape(s string) string {
	return strings.NewReplacer("*", `\*`, "~", `\~`, "`", "\\`", "||", `\|\|`).Replace(s)
}

// collapse trims blank lines at the ends and squeezes runs of blank lines to one.
func collapse(lines []string) string {
	var out []string
	blank := true
	for _, l := range lines {
		if strings.TrimSpace(l) == "" {
			if !blank {
				out = append(out, "")
			}
			blank = true
			continue
		}
		out = append(out, l)
		blank = false
	}
	for len(out) > 0 && out[len(out)-1] == "" {
		out = out[:len(out)-1]
	}
	return strings.Join(out, "\n")
}
