// pankhllm C++17 client, header-only over libcurl. JSON is passed through as
// std::string so you can use nlohmann::json or any other library to build/parse it.
//   g++ -std=c++17 app.cpp -lcurl
#pragma once
#include <curl/curl.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pankhllm {

struct Response { long status = 0; std::string body; };

class Error : public std::runtime_error {
public:
    long status;
    Error(long s, const std::string& m) : std::runtime_error(std::to_string(s) + ": " + m), status(s) {}
};

class Client {
public:
    explicit Client(std::string base_url = "http://localhost:4000", std::vector<std::string> headers = {})
        : base_(std::move(base_url)), headers_(std::move(headers)) {
        while (!base_.empty() && base_.back() == '/') base_.pop_back();
    }

    /// POST an OpenAI-shaped chat body (with optional "pankhllm" extension). Returns raw JSON.
    std::string chat(const std::string& body_json) { return post("/v1/chat/completions", body_json, nullptr).body; }

    /// Streaming: body_json must contain "stream": true. on_delta receives each text fragment.
    void chat_stream(const std::string& body_json, const std::function<void(const std::string&)>& on_delta) {
        post("/v1/chat/completions", body_json, &on_delta);
    }

    /// Dry run: routing decision only.
    std::string route(const std::string& body_json) { return post("/v1/route", body_json, nullptr).body; }

    /// Convenience: minimal body for a single question, with optional context JSON array and tags JSON array.
    static std::string simple_body(const std::string& question, const std::string& context_json = "", const std::string& tags_json = "", bool stream = false) {
        std::string ext;
        if (!context_json.empty()) ext += "\"context\":" + context_json;
        if (!tags_json.empty()) ext += (ext.empty() ? "" : ",") + std::string("\"tags\":") + tags_json;
        return "{\"model\":\"auto\",\"stream\":" + std::string(stream ? "true" : "false") +
               ",\"messages\":[{\"role\":\"user\",\"content\":\"" + escape(question) + "\"}],\"pankhllm\":{" + ext + "}}";
    }

private:
    std::string base_;
    std::vector<std::string> headers_;

    static std::string escape(const std::string& s) {
        std::string o;
        for (char c : s) switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default: if (static_cast<unsigned char>(c) < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += c;
        }
        return o;
    }

    struct Ctx { std::string all, line; const std::function<void(const std::string&)>* cb; };

    static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* user) {
        auto* c = static_cast<Ctx*>(user);
        size_t n = size * nmemb;
        c->all.append(ptr, n);
        if (!c->cb) return n;
        c->line.append(ptr, n);
        size_t pos;
        while ((pos = c->line.find('\n')) != std::string::npos) {
            std::string l = c->line.substr(0, pos);
            c->line.erase(0, pos + 1);
            if (l.rfind("data:", 0) != 0) continue;
            std::string d = l.substr(5);
            d.erase(0, d.find_first_not_of(' '));
            if (d == "[DONE]") continue;
            auto k = d.find("\"delta\":{");
            if (k == std::string::npos) continue;
            k = d.find("\"content\":\"", k);
            if (k == std::string::npos) continue;
            k += 11;
            std::string out;
            for (; k < d.size() && d[k] != '"'; ++k) {
                if (d[k] == '\\' && k + 1 < d.size()) {
                    ++k;
                    switch (d[k]) { case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break;
                                    case 'u': { unsigned v = std::stoul(d.substr(k + 1, 4), nullptr, 16); out += v < 0x80 ? static_cast<char>(v) : '?'; k += 4; break; }
                                    default: out += d[k]; }
                } else out += d[k];
            }
            if (!out.empty()) (*c->cb)(out);
        }
        return n;
    }

    Response post(const std::string& path, const std::string& body, const std::function<void(const std::string&)>* cb) {
        CURL* c = curl_easy_init();
        if (!c) throw Error(0, "curl init failed");
        struct curl_slist* h = curl_slist_append(nullptr, "Content-Type: application/json");
        for (auto& x : headers_) h = curl_slist_append(h, x.c_str());
        Ctx ctx{"", "", cb};
        std::string url = base_ + path;
        curl_easy_setopt(c, CURLOPT_URL, url.c_str());
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
        curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(c, CURLOPT_TIMEOUT, 300L);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, &ctx);
        CURLcode rc = curl_easy_perform(c);
        Response r;
        if (rc == CURLE_OK) curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
        curl_slist_free_all(h);
        curl_easy_cleanup(c);
        if (rc != CURLE_OK) throw Error(0, curl_easy_strerror(rc));
        r.body = std::move(ctx.all);
        if (r.status >= 400) throw Error(r.status, r.body);
        return r;
    }
};

}  // namespace pankhllm
