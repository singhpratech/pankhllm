package dev.pankhllm;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * pankhllm client for Java 17+ using only the JDK HTTP client. JSON is built and
 * read with a tiny embedded encoder/decoder so the SDK has no dependencies;
 * swap in Jackson or Gson freely in your own code.
 */
public final class PankhClient {
    public record Chunk(String text, Double score, String source) {}
    public record Message(String role, String content) {}

    public static final class Options {
        public List<Message> messages;
        public List<Chunk> context;
        public List<String> tags;
        public String prompt, tier, intent, model;
        public Integer maxTokens, maxLatencyMs;
        public Double maxCostUsd, temperature;
    }

    public record Answer(String text, String model, String providerModel, double costUsd, boolean clarification, double confidence, String rawJson) {}

    public static final class PankhException extends RuntimeException {
        public final int status;
        public PankhException(int status, String message) { super(status + ": " + message); this.status = status; }
    }

    private final String base;
    private final HttpClient http;
    private final Map<String, String> headers;

    public PankhClient(String baseUrl) { this(baseUrl, Map.of()); }

    public PankhClient(String baseUrl, Map<String, String> headers) {
        this.base = baseUrl.replaceAll("/+$", "");
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        this.headers = headers;
    }

    private String body(String question, Options o, boolean stream) {
        StringBuilder ext = new StringBuilder("{");
        if (o.context != null && !o.context.isEmpty()) {
            ext.append("\"context\":[");
            for (int i = 0; i < o.context.size(); i++) {
                Chunk c = o.context.get(i);
                if (i > 0) ext.append(',');
                ext.append("{\"text\":").append(Json.str(c.text()));
                if (c.score() != null) ext.append(",\"score\":").append(c.score());
                if (c.source() != null) ext.append(",\"source\":").append(Json.str(c.source()));
                ext.append('}');
            }
            ext.append("],");
        }
        if (o.tags != null && !o.tags.isEmpty()) ext.append("\"tags\":").append(Json.strList(o.tags)).append(',');
        if (o.prompt != null) ext.append("\"prompt\":").append(Json.str(o.prompt)).append(',');
        if (o.tier != null) ext.append("\"tier\":").append(Json.str(o.tier)).append(',');
        if (o.intent != null) ext.append("\"intent\":").append(Json.str(o.intent)).append(',');
        if (o.maxCostUsd != null) ext.append("\"max_cost_usd\":").append(o.maxCostUsd).append(',');
        if (o.maxLatencyMs != null) ext.append("\"max_latency_ms\":").append(o.maxLatencyMs).append(',');
        if (ext.charAt(ext.length() - 1) == ',') ext.setLength(ext.length() - 1);
        ext.append('}');

        StringBuilder msgs = new StringBuilder("[");
        List<Message> ms = o.messages != null ? o.messages : List.of(new Message("user", question == null ? "" : question));
        for (int i = 0; i < ms.size(); i++) {
            if (i > 0) msgs.append(',');
            msgs.append("{\"role\":").append(Json.str(ms.get(i).role())).append(",\"content\":").append(Json.str(ms.get(i).content())).append('}');
        }
        msgs.append(']');

        StringBuilder b = new StringBuilder("{\"model\":").append(Json.str(o.model != null ? o.model : "auto"))
            .append(",\"messages\":").append(msgs).append(",\"stream\":").append(stream).append(",\"pankhllm\":").append(ext);
        if (o.maxTokens != null) b.append(",\"max_tokens\":").append(o.maxTokens);
        if (o.temperature != null) b.append(",\"temperature\":").append(o.temperature);
        return b.append('}').toString();
    }

    private HttpRequest request(String path, String json) {
        HttpRequest.Builder r = HttpRequest.newBuilder(URI.create(base + path))
            .header("Content-Type", "application/json").timeout(Duration.ofMinutes(2))
            .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8));
        headers.forEach(r::header);
        return r.build();
    }

    /** Route and answer. */
    public Answer ask(String question, Options o) throws Exception {
        if (o == null) o = new Options();
        HttpResponse<String> resp = http.send(request("/v1/chat/completions", body(question, o, false)), HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) throw new PankhException(resp.statusCode(), resp.body());
        String raw = resp.body();
        String text = Json.find(raw, "\"content\":");
        double conf = Json.number(raw, "\"score\":");
        return new Answer(text, Json.findAfter(raw, "\"pankhllm\":", "\"model\":"), Json.find(raw, "\"model\":"), Json.number(raw, "\"cost_usd\":"),
            raw.contains("\"clarification\":true"), conf, raw);
    }

    /** Stream text deltas to the consumer. Returns when the stream ends. */
    public void stream(String question, Options o, Consumer<String> onDelta) throws Exception {
        if (o == null) o = new Options();
        HttpResponse<java.io.InputStream> resp = http.send(request("/v1/chat/completions", body(question, o, true)), HttpResponse.BodyHandlers.ofInputStream());
        if (resp.statusCode() >= 400) throw new PankhException(resp.statusCode(), new String(resp.body().readAllBytes(), StandardCharsets.UTF_8));
        try (BufferedReader r = new BufferedReader(new InputStreamReader(resp.body(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) {
                if (!line.startsWith("data:")) continue;
                String data = line.substring(5).trim();
                if (data.equals("[DONE]")) return;
                if (data.contains("\"finish_reason\":\"error\"")) throw new PankhException(502, data);
                int i = data.indexOf("\"content\":\"");
                if (i >= 0) {
                    String delta = Json.readString(data, i + "\"content\":".length());
                    if (!delta.isEmpty()) onDelta.accept(delta);
                }
            }
        }
    }

    /** Dry run: the routing decision as raw JSON. */
    public String route(String question, Options o) throws Exception {
        if (o == null) o = new Options();
        HttpResponse<String> resp = http.send(request("/v1/route", body(question, o, false)), HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) throw new PankhException(resp.statusCode(), resp.body());
        return resp.body();
    }

    /** Minimal JSON helpers: enough for this wire format, not a general parser. */
    static final class Json {
        static String str(String s) {
            StringBuilder b = new StringBuilder("\"");
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"' -> b.append("\\\"");
                    case '\\' -> b.append("\\\\");
                    case '\n' -> b.append("\\n");
                    case '\r' -> b.append("\\r");
                    case '\t' -> b.append("\\t");
                    default -> { if (c < 0x20) b.append(String.format("\\u%04x", (int) c)); else b.append(c); }
                }
            }
            return b.append('"').toString();
        }
        static String strList(List<String> xs) {
            List<String> q = new ArrayList<>();
            for (String x : xs) q.add(str(x));
            return "[" + String.join(",", q) + "]";
        }
        static String find(String json, String key) {
            int i = json.indexOf(key);
            return i < 0 ? "" : readString(json, i + key.length());
        }
        static String findAfter(String json, String anchor, String key) {
            int a = json.indexOf(anchor);
            if (a < 0) return "";
            int i = json.indexOf(key, a);
            return i < 0 ? "" : readString(json, i + key.length());
        }
        static double number(String json, String key) {
            int i = json.indexOf(key);
            if (i < 0) return 0;
            int j = i + key.length();
            int k = j;
            while (k < json.length() && "0123456789.-eE".indexOf(json.charAt(k)) >= 0) k++;
            try { return Double.parseDouble(json.substring(j, k)); } catch (NumberFormatException e) { return 0; }
        }
        static String readString(String json, int at) {
            while (at < json.length() && json.charAt(at) == ' ') at++;
            if (at >= json.length() || json.charAt(at) != '"') return "";
            StringBuilder b = new StringBuilder();
            for (int i = at + 1; i < json.length(); i++) {
                char c = json.charAt(i);
                if (c == '\\') {
                    char n = json.charAt(++i);
                    switch (n) {
                        case 'n' -> b.append('\n');
                        case 'r' -> b.append('\r');
                        case 't' -> b.append('\t');
                        case 'u' -> { b.append((char) Integer.parseInt(json.substring(i + 1, i + 5), 16)); i += 4; }
                        default -> b.append(n);
                    }
                } else if (c == '"') return b.toString();
                else b.append(c);
            }
            return b.toString();
        }
    }
}
