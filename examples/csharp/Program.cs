// pankhllm from .NET: plain HttpClient, Semantic Kernel, and a SignalR hub that streams.
// No pankhllm-specific package is needed; the server speaks the OpenAI wire format.

using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.AspNetCore.SignalR;

// ---------- 1. Semantic Kernel: point the OpenAI connector at the router ----------
// Routing hints that SK cannot put in the body go in headers, set once on the HttpClient.
var http = new HttpClient { BaseAddress = new Uri("http://localhost:4000/v1/") };
http.DefaultRequestHeaders.Add("X-Pankh-Tags", "private");          // only models tagged private
http.DefaultRequestHeaders.Add("X-Pankh-Prompt", "support-agent");  // server-held skill/agent prompt
http.DefaultRequestHeaders.Add("X-Pankh-Max-Latency-Ms", "8000");   // hard wall-clock budget

var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(modelId: "auto", apiKey: "unused", httpClient: http)   // "auto", "auto/fast", or a model name
    .Build();

var chat = kernel.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();
history.AddUserMessage("What is the refund window for annual plans?");
var reply = await chat.GetChatMessageContentAsync(history);
Console.WriteLine(reply.Content);

// ---------- 2. Direct call with retrieved context (full control) ----------
var body = new
{
    model = "auto",
    messages = new[] { new { role = "user", content = "Why did the migration fail?" } },
    pankhllm = new
    {
        context = new[] { new { text = "Postmortem: the migration failed because ...", score = 0.82, source = "pm-2024-03.md" } },
        tags = new[] { "private" },
        max_latency_ms = 8000,
    },
};
var resp = await http.PostAsJsonAsync("chat/completions", body);
using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
Console.WriteLine(doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
Console.WriteLine("routed to " + doc.RootElement.GetProperty("pankhllm").GetProperty("model").GetString());

// ---------- 3. SignalR hub that streams tokens to the browser ----------
public class ChatHub : Hub
{
    private readonly HttpClient _http;
    public ChatHub(IHttpClientFactory f) => _http = f.CreateClient("pankhllm");

    public async IAsyncEnumerable<string> Ask(string question, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, "chat/completions")
        {
            Content = JsonContent.Create(new { model = "auto", stream = true, messages = new[] { new { role = "user", content = question } } })
        };
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
        resp.EnsureSuccessStatusCode();
        using var reader = new StreamReader(await resp.Content.ReadAsStreamAsync(ct));
        while (!reader.EndOfStream && !ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (line is null || !line.StartsWith("data:")) continue;
            var data = line[5..].Trim();
            if (data == "[DONE]") yield break;
            using var ev = JsonDocument.Parse(data);
            var delta = ev.RootElement.GetProperty("choices")[0].GetProperty("delta");
            if (delta.TryGetProperty("content", out var c) && c.GetString() is { Length: > 0 } text)
                yield return text;
        }
    }
}
