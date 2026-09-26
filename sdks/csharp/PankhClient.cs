// pankhllm client for .NET 6+. No dependencies beyond System.Net.Http and System.Text.Json.
// For Semantic Kernel / Microsoft Agent Framework you do not need this class: point the
// OpenAI connector at http://<router>/v1 and add X-Pankh-* headers on the HttpClient.
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Pankhllm;

public record Chunk(string Text, double? Score = null, string? Source = null);
public record ChatMessage(string Role, string Content);

public sealed class AskOptions
{
    public IList<ChatMessage>? Messages { get; set; }
    public IList<Chunk>? Context { get; set; }
    public IList<string>? Tags { get; set; }
    public string? Prompt { get; set; }
    public string? Tier { get; set; }      // fast | balanced | reasoning
    public string? Intent { get; set; }
    public string? Model { get; set; }     // "auto" (default), "auto/fast", or a configured model name
    public int? MaxTokens { get; set; }
    public double? MaxCostUsd { get; set; }
    public int? MaxLatencyMs { get; set; }
    public double? Temperature { get; set; }
}

public sealed class Answer
{
    public string Text { get; init; } = "";
    public string Model { get; init; } = "";
    public string ProviderModel { get; init; } = "";
    public double CostUsd { get; init; }
    public bool Abstained { get; init; }
    public bool Clarification { get; init; }
    public double Confidence { get; init; }
    public JsonElement Decision { get; init; }
    public JsonElement Attempts { get; init; }
    public JsonElement Raw { get; init; }
}

public sealed class PankhException : Exception
{
    public int Status { get; }
    public string Kind { get; }
    public PankhException(int status, string kind, string message) : base($"{status} {kind}: {message}") { Status = status; Kind = kind; }
}

public sealed class PankhClient
{
    private readonly HttpClient _http;

    public PankhClient(string baseUrl = "http://localhost:4000", HttpClient? http = null)
    {
        _http = http ?? new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        _http.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
    }

    private static Dictionary<string, object?> Body(string? question, AskOptions o, bool stream)
    {
        var ext = new Dictionary<string, object?>();
        if (o.Context is { Count: > 0 }) ext["context"] = o.Context.Select(c => new Dictionary<string, object?> { ["text"] = c.Text, ["score"] = c.Score, ["source"] = c.Source });
        if (o.Tags is { Count: > 0 }) ext["tags"] = o.Tags;
        if (o.Prompt is not null) ext["prompt"] = o.Prompt;
        if (o.Tier is not null) ext["tier"] = o.Tier;
        if (o.Intent is not null) ext["intent"] = o.Intent;
        if (o.MaxCostUsd is not null) ext["max_cost_usd"] = o.MaxCostUsd;
        if (o.MaxLatencyMs is not null) ext["max_latency_ms"] = o.MaxLatencyMs;
        var body = new Dictionary<string, object?>
        {
            ["model"] = o.Model ?? "auto",
            ["messages"] = o.Messages?.Select(m => new { role = m.Role, content = m.Content }).ToList<object>() ?? new List<object> { new { role = "user", content = question ?? "" } },
            ["stream"] = stream,
            ["pankhllm"] = ext,
        };
        if (o.MaxTokens is not null) body["max_tokens"] = o.MaxTokens;
        if (o.Temperature is not null) body["temperature"] = o.Temperature;
        return body;
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };

    private async Task<HttpResponseMessage> PostAsync(string path, object body, bool streaming, CancellationToken ct)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, path) { Content = JsonContent.Create(body, options: JsonOpts) };
        var resp = await _http.SendAsync(req, streaming ? HttpCompletionOption.ResponseHeadersRead : HttpCompletionOption.ResponseContentRead, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var text = await resp.Content.ReadAsStringAsync(ct);
            try
            {
                var e = JsonDocument.Parse(text).RootElement.GetProperty("error");
                throw new PankhException((int)resp.StatusCode, e.GetProperty("type").GetString() ?? "error", e.GetProperty("message").GetString() ?? text);
            }
            catch (JsonException) { throw new PankhException((int)resp.StatusCode, "error", text); }
            catch (KeyNotFoundException) { throw new PankhException((int)resp.StatusCode, "error", text); }
        }
        return resp;
    }

    public async Task<Answer> AskAsync(string? question, AskOptions? options = null, CancellationToken ct = default)
    {
        var o = options ?? new AskOptions();
        using var resp = await PostAsync("v1/chat/completions", Body(question, o, false), false, ct);
        var v = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct)).RootElement;
        var p = v.TryGetProperty("pankhllm", out var pk) ? pk : default;
        return new Answer
        {
            Text = v.GetProperty("choices")[0].GetProperty("message").TryGetProperty("content", out var c) && c.ValueKind == JsonValueKind.String ? c.GetString()! : "",
            Model = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("model", out var m) ? m.GetString() ?? "" : "",
            ProviderModel = v.TryGetProperty("model", out var pm) ? pm.GetString() ?? "" : "",
            CostUsd = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("cost_usd", out var cost) ? cost.GetDouble() : 0,
            Abstained = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("abstained", out var ab) && ab.GetBoolean(),
            Clarification = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("clarification", out var cl) && cl.GetBoolean(),
            Confidence = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("confidence", out var cf) && cf.TryGetProperty("score", out var sc) ? sc.GetDouble() : 0,
            Decision = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("decision", out var d) ? d.Clone() : default,
            Attempts = p.ValueKind == JsonValueKind.Object && p.TryGetProperty("attempts", out var a) ? a.Clone() : default,
            Raw = v.Clone(),
        };
    }

    /// Streams text deltas. Routing metadata arrives in the first event's "pankhllm" field; use onMeta to capture it.
    public async IAsyncEnumerable<string> StreamAsync(string? question, AskOptions? options = null, Action<JsonElement>? onMeta = null, [EnumeratorCancellation] CancellationToken ct = default)
    {
        var o = options ?? new AskOptions();
        using var resp = await PostAsync("v1/chat/completions", Body(question, o, true), true, ct);
        using var reader = new StreamReader(await resp.Content.ReadAsStreamAsync(ct));
        var metaSent = false;
        while (!reader.EndOfStream && !ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (line is null || !line.StartsWith("data:")) continue;
            var data = line[5..].Trim();
            if (data == "[DONE]") yield break;
            using var ev = JsonDocument.Parse(data);
            var root = ev.RootElement;
            if (!metaSent && root.TryGetProperty("pankhllm", out var meta)) { metaSent = true; onMeta?.Invoke(meta.Clone()); }
            if (!root.TryGetProperty("choices", out var choices)) continue;
            foreach (var ch in choices.EnumerateArray())
            {
                if (ch.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String && fr.GetString() == "error")
                    throw new PankhException(502, "upstream_error", root.TryGetProperty("pankhllm", out var pe) ? pe.ToString() : "stream error");
                if (ch.TryGetProperty("delta", out var delta) && delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String && content.GetString() is { Length: > 0 } text)
                    yield return text;
            }
        }
    }

    /// Dry run: the routing decision, no model call.
    public async Task<JsonElement> RouteAsync(string? question, AskOptions? options = null, CancellationToken ct = default)
    {
        using var resp = await PostAsync("v1/route", Body(question, options ?? new AskOptions(), false), false, ct);
        return JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct)).RootElement.Clone();
    }
}
