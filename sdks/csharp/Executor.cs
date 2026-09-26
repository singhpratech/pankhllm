// Planner-lane executor for ASP.NET Core minimal APIs.
//
//   app.MapPankhExecutor("/execute", new Dictionary<string, PankhOp> {
//       ["metric_by_period"] = async (p, ctx) => new { value = await Kpi(p["metric"]!.GetValue<string>(), ctx.User) },
//   });
//
// The router POSTs { op, params, question, user, tenant, trace_id } after validating the plan
// against your catalog; return data (rendered by the operation's template), a string answer,
// or a list of rows. Exceptions become { ok = false } so the router falls through to the agent.
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace Pankhllm;

public sealed record PlanContext(string? Question, string? User, string? Tenant, string? TraceId);
public delegate Task<object?> PankhOp(JsonObject parameters, PlanContext ctx);

public static class PankhExecutor
{
    public static async Task<object> HandleAsync(JsonNode? payload, IReadOnlyDictionary<string, PankhOp> ops)
    {
        var op = payload?["op"]?.GetValue<string>();
        if (op is null || !ops.TryGetValue(op, out var fn)) return new { ok = false, error = $"unknown operation {op}" };
        var ctx = new PlanContext(payload?["question"]?.GetValue<string>(), payload?["user"]?.GetValue<string>(), payload?["tenant"]?.GetValue<string>(), payload?["trace_id"]?.GetValue<string>());
        try
        {
            var p = payload?["params"] as JsonObject ?? new JsonObject();
            var result = await fn(p, ctx);
            return result switch
            {
                string s => new { ok = true, answer = s },
                System.Collections.IEnumerable rows and not string => new { ok = true, data = new { rows } },
                _ => new { ok = true, data = result },
            };
        }
        catch (Exception e)
        {
            return new { ok = false, error = $"{e.GetType().Name}: {e.Message}" };
        }
    }

    public static RouteHandlerBuilder MapPankhExecutor(this IEndpointRouteBuilder app, string path, IReadOnlyDictionary<string, PankhOp> ops, string? apiKey = null) =>
        app.MapPost(path, async (HttpContext http) =>
        {
            if (apiKey is not null && http.Request.Headers.Authorization != $"Bearer {apiKey}")
                return Results.Json(new { ok = false, error = "unauthorized" }, statusCode: 401);
            var payload = await JsonNode.ParseAsync(http.Request.Body);
            return Results.Json(await HandleAsync(payload, ops));
        });
}
