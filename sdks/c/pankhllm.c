#include "pankhllm.h"
#include <curl/curl.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { char *data; size_t len; } buf_t;

static size_t collect(void *ptr, size_t size, size_t nmemb, void *user) {
    buf_t *b = user; size_t n = size * nmemb;
    char *d = realloc(b->data, b->len + n + 1);
    if (!d) return 0;
    b->data = d; memcpy(b->data + b->len, ptr, n); b->len += n; b->data[b->len] = 0;
    return n;
}

typedef struct { buf_t line; pankh_delta_cb cb; void *user; buf_t all; } sse_t;

/* Extract "content":"..." from a chunk line and hand the unescaped text to the callback. */
static void emit_delta(sse_t *s, const char *data) {
    const char *k = strstr(data, "\"delta\":{");
    if (!k) return;
    k = strstr(k, "\"content\":\"");
    if (!k) return;
    k += 11;
    char *out = malloc(strlen(k) + 1); size_t o = 0;
    for (; *k && *k != '"'; k++) {
        if (*k == '\\' && k[1]) {
            k++;
            switch (*k) { case 'n': out[o++] = '\n'; break; case 't': out[o++] = '\t'; break; case 'r': out[o++] = '\r'; break;
                          case 'u': { unsigned c = 0; sscanf(k + 1, "%4x", &c); if (c < 0x80) out[o++] = (char)c; else { out[o++] = '?'; } k += 4; break; }
                          default: out[o++] = *k; }
        } else out[o++] = *k;
    }
    if (o) s->cb(out, o, s->user);
    free(out);
}

static size_t sse_write(void *ptr, size_t size, size_t nmemb, void *user) {
    sse_t *s = user; size_t n = size * nmemb;
    collect(ptr, size, nmemb, &s->all);
    if (collect(ptr, size, nmemb, &s->line) == 0) return 0;
    char *start = s->line.data, *nl;
    while ((nl = strchr(start, '\n'))) {
        *nl = 0;
        if (strncmp(start, "data:", 5) == 0) {
            const char *d = start + 5; while (*d == ' ') d++;
            if (strcmp(d, "[DONE]") != 0) emit_delta(s, d);
        }
        start = nl + 1;
    }
    size_t rest = strlen(start);
    memmove(s->line.data, start, rest + 1); s->line.len = rest;
    return n;
}

static pankh_response do_post(const char *base_url, const char *path, const char *body, const char **extra, pankh_delta_cb cb, void *user) {
    pankh_response r = {0, NULL};
    CURL *c = curl_easy_init();
    if (!c) { r.body = strdup("curl init failed"); return r; }
    char url[2048]; snprintf(url, sizeof url, "%s%s", base_url, path);
    struct curl_slist *h = curl_slist_append(NULL, "Content-Type: application/json");
    for (; extra && *extra; extra++) h = curl_slist_append(h, *extra);
    buf_t plain = {NULL, 0}; sse_t sse = {{NULL, 0}, cb, user, {NULL, 0}};
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, h);
    curl_easy_setopt(c, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, 300L);
    if (cb) { curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, sse_write); curl_easy_setopt(c, CURLOPT_WRITEDATA, &sse); }
    else { curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, collect); curl_easy_setopt(c, CURLOPT_WRITEDATA, &plain); }
    CURLcode rc = curl_easy_perform(c);
    if (rc != CURLE_OK) { r.body = strdup(curl_easy_strerror(rc)); }
    else { curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status); r.body = cb ? sse.all.data : plain.data; if (!r.body) r.body = strdup(""); }
    if (cb) { free(sse.line.data); if (rc != CURLE_OK) free(sse.all.data); } else if (rc != CURLE_OK) free(plain.data);
    curl_slist_free_all(h); curl_easy_cleanup(c);
    return r;
}

pankh_response pankh_chat(const char *b, const char *body, const char **x) { return do_post(b, "/v1/chat/completions", body, x, NULL, NULL); }
pankh_response pankh_chat_stream(const char *b, const char *body, const char **x, pankh_delta_cb cb, void *u) { return do_post(b, "/v1/chat/completions", body, x, cb, u); }
pankh_response pankh_route(const char *b, const char *body, const char **x) { return do_post(b, "/v1/route", body, x, NULL, NULL); }

static void json_escape(const char *s, char *out) {
    for (; *s; s++) switch (*s) {
        case '"': *out++ = '\\'; *out++ = '"'; break; case '\\': *out++ = '\\'; *out++ = '\\'; break;
        case '\n': *out++ = '\\'; *out++ = 'n'; break; case '\r': *out++ = '\\'; *out++ = 'r'; break; case '\t': *out++ = '\\'; *out++ = 't'; break;
        default: if ((unsigned char)*s < 0x20) out += sprintf(out, "\\u%04x", *s); else *out++ = *s; }
    *out = 0;
}

char *pankh_simple_body(const char *q, const char *ctx, const char *tags, int stream) {
    char *esc = malloc(strlen(q) * 6 + 1); json_escape(q, esc);
    size_t n = strlen(esc) + (ctx ? strlen(ctx) : 0) + (tags ? strlen(tags) : 0) + 200;
    char *b = malloc(n);
    snprintf(b, n, "{\"model\":\"auto\",\"stream\":%s,\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}],\"pankhllm\":{%s%s%s%s}}",
             stream ? "true" : "false", esc, ctx ? "\"context\":" : "", ctx ? ctx : "", (ctx && tags) ? "," : "", tags ? "\"tags\":" : "");
    if (tags) { size_t l = strlen(b); b[l - 2] = 0; snprintf(b + l - 2, n - l + 2, "%s}}", tags); }
    free(esc);
    return b;
}

void pankh_free(void *p) { free(p); }
