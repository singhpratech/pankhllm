// Package pankhllm is a client for pankhllm, the LLM gateway that learns to skip the LLM.
// The server is OpenAI-compatible, so any OpenAI Go client also works by
// changing its base URL; this package adds the routing extras.
package pankhllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Chunk struct {
	Text   string   `json:"text"`
	Score  *float64 `json:"score,omitempty"`
	Source string   `json:"source,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Options struct {
	Messages     []Message
	Context      []Chunk
	Tags         []string
	Prompt       string
	Tier         string // fast | balanced | reasoning
	Intent       string
	Model        string
	MaxTokens    int
	MaxCostUSD   *float64
	MaxLatencyMs int
	Temperature  *float64
}

type Answer struct {
	Text          string
	Model         string
	ProviderModel string
	CostUSD       float64
	Decision      json.RawMessage
	Attempts      json.RawMessage
	Abstained     bool
}

type Error struct {
	Status  int
	Kind    string
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("%d %s: %s", e.Status, e.Kind, e.Message) }

type Client struct {
	BaseURL string
	HTTP    *http.Client
	Headers map[string]string
}

func New(baseURL string) *Client {
	return &Client{BaseURL: strings.TrimRight(baseURL, "/"), HTTP: &http.Client{Timeout: 2 * time.Minute}}
}

func (c *Client) body(question string, o Options, stream bool) map[string]any {
	ext := map[string]any{}
	if len(o.Context) > 0 {
		ext["context"] = o.Context
	}
	if len(o.Tags) > 0 {
		ext["tags"] = o.Tags
	}
	if o.Prompt != "" {
		ext["prompt"] = o.Prompt
	}
	if o.Tier != "" {
		ext["tier"] = o.Tier
	}
	if o.Intent != "" {
		ext["intent"] = o.Intent
	}
	if o.MaxCostUSD != nil {
		ext["max_cost_usd"] = *o.MaxCostUSD
	}
	if o.MaxLatencyMs > 0 {
		ext["max_latency_ms"] = o.MaxLatencyMs
	}
	msgs := o.Messages
	if len(msgs) == 0 {
		msgs = []Message{{Role: "user", Content: question}}
	}
	model := o.Model
	if model == "" {
		model = "auto"
	}
	b := map[string]any{"model": model, "messages": msgs, "stream": stream, "pankhllm": ext}
	if o.MaxTokens > 0 {
		b["max_tokens"] = o.MaxTokens
	}
	if o.Temperature != nil {
		b["temperature"] = *o.Temperature
	}
	return b
}

func (c *Client) post(ctx context.Context, path string, body any) (*http.Response, error) {
	raw, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.Headers {
		req.Header.Set(k, v)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		var e struct {
			Error struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			} `json:"error"`
		}
		if json.Unmarshal(data, &e) == nil && e.Error.Message != "" {
			return nil, &Error{Status: resp.StatusCode, Kind: e.Error.Type, Message: e.Error.Message}
		}
		return nil, &Error{Status: resp.StatusCode, Kind: "error", Message: string(data)}
	}
	return resp, nil
}

// Ask routes the question and returns the answer plus the routing decision.
func (c *Client) Ask(ctx context.Context, question string, o Options) (*Answer, error) {
	resp, err := c.post(ctx, "/v1/chat/completions", c.body(question, o, false))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var v struct {
		Model   string `json:"model"`
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
		Pankh struct {
			Model     string          `json:"model"`
			CostUSD   float64         `json:"cost_usd"`
			Abstained bool            `json:"abstained"`
			Decision  json.RawMessage `json:"decision"`
			Attempts  json.RawMessage `json:"attempts"`
		} `json:"pankhllm"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		return nil, err
	}
	a := &Answer{Model: v.Pankh.Model, ProviderModel: v.Model, CostUSD: v.Pankh.CostUSD, Decision: v.Pankh.Decision, Attempts: v.Pankh.Attempts, Abstained: v.Pankh.Abstained}
	if len(v.Choices) > 0 {
		a.Text = v.Choices[0].Message.Content
	}
	return a, nil
}

// Stream calls onDelta for each text fragment. onMeta (optional) receives routing metadata first.
func (c *Client) Stream(ctx context.Context, question string, o Options, onMeta func(json.RawMessage), onDelta func(string)) error {
	resp, err := c.post(ctx, "/v1/chat/completions", c.body(question, o, true))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	metaSent := false
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[5:])
		if data == "[DONE]" {
			return nil
		}
		var ev struct {
			Pankh   json.RawMessage `json:"pankhllm"`
			Choices []struct {
				Delta        struct{ Content string `json:"content"` } `json:"delta"`
				FinishReason string                                     `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return err
		}
		if !metaSent && len(ev.Pankh) > 0 && onMeta != nil {
			metaSent = true
			onMeta(ev.Pankh)
		}
		for _, ch := range ev.Choices {
			if ch.FinishReason == "error" {
				return &Error{Status: 502, Kind: "upstream_error", Message: string(ev.Pankh)}
			}
			if ch.Delta.Content != "" {
				onDelta(ch.Delta.Content)
			}
		}
	}
	return sc.Err()
}

// Route is a dry run: the decision without calling any model.
func (c *Client) Route(ctx context.Context, question string, o Options) (json.RawMessage, error) {
	resp, err := c.post(ctx, "/v1/route", c.body(question, o, false))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}
