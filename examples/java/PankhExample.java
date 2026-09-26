// pankhllm from Java with the JDK HTTP client only. Any OpenAI-compatible Java
// SDK (LangChain4j, Spring AI, openai-java) also works by setting the base URL.

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class PankhExample {
    public static void main(String[] args) throws Exception {
        HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

        String body = """
            {
              "model": "auto",
              "messages": [{"role": "user", "content": "Compare the enterprise and team plans"}],
              "pankhllm": {
                "context": [{"text": "Enterprise: SSO, 99.9% SLA ...", "score": 0.77, "source": "pricing.md"}],
                "max_latency_ms": 8000,
                "max_cost_usd": 0.02
              }
            }
            """;

        HttpRequest req = HttpRequest.newBuilder(URI.create("http://localhost:4000/v1/chat/completions"))
            .header("Content-Type", "application/json")
            .header("X-Pankh-Prompt", "support-agent")
            .timeout(Duration.ofSeconds(30))
            .POST(HttpRequest.BodyPublishers.ofString(body))
            .build();

        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        System.out.println(resp.statusCode());
        System.out.println(resp.body()); // choices[0].message.content plus pankhllm.{model,decision,attempts}
    }
}
