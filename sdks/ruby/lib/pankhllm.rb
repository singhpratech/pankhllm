# frozen_string_literal: true
# pankhllm client for Ruby 3+. Standard library only (net/http, json).
require "net/http"
require "json"
require "uri"

module Pankhllm
  class Error < StandardError
    attr_reader :status, :kind
    def initialize(status, kind, message)
      @status, @kind = status, kind
      super("#{status} #{kind}: #{message}")
    end
  end

  Answer = Struct.new(:text, :model, :provider_model, :cost_usd, :abstained, :clarification, :confidence, :decision, :attempts, :raw, keyword_init: true)

  class Client
    def initialize(base_url = "http://localhost:4000", headers: {}, timeout: 120)
      @base = URI(base_url.sub(%r{/+$}, ""))
      @headers = headers
      @timeout = timeout
    end

    # ask("question", context: [{text:, score:, source:}], tags: ["private"], prompt: "skill", tier: "fast",
    #     intent: "lookup", model: "auto", max_tokens: 400, max_cost_usd: 0.01, max_latency_ms: 8000)
    def ask(question = nil, **o)
      v = post("/v1/chat/completions", body(question, o, false))
      p = v["pankhllm"] || {}
      Answer.new(text: v.dig("choices", 0, "message", "content").to_s, model: p["model"].to_s, provider_model: v["model"].to_s,
                 cost_usd: p["cost_usd"].to_f, abstained: !!p["abstained"], clarification: !!p["clarification"],
                 confidence: p.dig("confidence", "score").to_f, decision: p["decision"] || {}, attempts: p["attempts"] || [], raw: v)
    end

    # Yields text deltas. The routing metadata is yielded first as a Hash if a block accepts two args.
    def stream(question = nil, **o)
      req = request("/v1/chat/completions", body(question, o, true))
      meta_sent = false
      http.request(req) do |resp|
        raise_error(resp) if resp.code.to_i >= 400
        buffer = +""
        resp.read_body do |chunk|
          buffer << chunk
          while (idx = buffer.index("\n\n"))
            block = buffer.slice!(0, idx + 2)
            block.each_line do |line|
              next unless line.start_with?("data:")
              data = line[5..].strip
              return if data == "[DONE]"
              ev = JSON.parse(data)
              if !meta_sent && ev["pankhllm"]
                meta_sent = true
                yield nil, ev["pankhllm"] if block_given?
              end
              (ev["choices"] || []).each do |ch|
                raise Error.new(502, "upstream_error", (ev["pankhllm"] || {})["error"].to_s) if ch["finish_reason"] == "error"
                delta = ch.dig("delta", "content")
                yield delta, nil if delta && !delta.empty? && block_given?
              end
            end
          end
        end
      end
    end

    # Dry run: routing decision only.
    def route(question = nil, **o)
      post("/v1/route", body(question, o, false))
    end

    def stats
      get("/v1/stats")
    end

    private

    def body(question, o, stream)
      ext = {}
      ext["context"] = o[:context] if o[:context]&.any?
      ext["tags"] = o[:tags] if o[:tags]&.any?
      { prompt: "prompt", tier: "tier", intent: "intent", max_cost_usd: "max_cost_usd", max_latency_ms: "max_latency_ms" }.each { |k, w| ext[w] = o[k] unless o[k].nil? }
      b = { "model" => o[:model] || "auto", "messages" => o[:messages] || [{ "role" => "user", "content" => question.to_s }], "stream" => stream, "pankhllm" => ext }
      b["max_tokens"] = o[:max_tokens] if o[:max_tokens]
      b["temperature"] = o[:temperature] if o[:temperature]
      b
    end

    def http
      h = Net::HTTP.new(@base.host, @base.port)
      h.use_ssl = @base.scheme == "https"
      h.read_timeout = @timeout
      h
    end

    def request(path, payload)
      req = Net::HTTP::Post.new(@base.path + path, { "Content-Type" => "application/json" }.merge(@headers))
      req.body = JSON.generate(payload)
      req
    end

    def post(path, payload)
      resp = http.request(request(path, payload))
      raise_error(resp) if resp.code.to_i >= 400
      JSON.parse(resp.body)
    end

    def get(path)
      resp = http.request(Net::HTTP::Get.new(@base.path + path, @headers))
      raise_error(resp) if resp.code.to_i >= 400
      JSON.parse(resp.body)
    end

    def raise_error(resp)
      body = resp.body.to_s
      begin
        e = JSON.parse(body)["error"] || {}
        raise Error.new(resp.code.to_i, e["type"] || "error", e["message"] || body)
      rescue JSON::ParserError
        raise Error.new(resp.code.to_i, "error", body)
      end
    end
  end
end
