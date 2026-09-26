/* pankhllm C client. Depends on libcurl only. The caller builds the JSON body
 * (or uses pankh_simple_body) and gets the raw JSON response back; parse it with
 * your JSON library of choice (cJSON, jansson, yyjson).
 *
 *   cc app.c pankhllm.c -lcurl
 */
#ifndef PANKHLLM_H
#define PANKHLLM_H
#include <stddef.h>

typedef struct {
    long status;      /* HTTP status; 0 on transport failure */
    char *body;       /* malloc'd response body (or error text); free with pankh_free */
} pankh_response;

typedef void (*pankh_delta_cb)(const char *delta, size_t len, void *user);

/* POST body_json to base_url + "/v1/chat/completions". extra_headers: NULL-terminated
 * array of "Name: value" strings (e.g. X-Pankh-Tags), or NULL. */
pankh_response pankh_chat(const char *base_url, const char *body_json, const char **extra_headers);

/* Same, but with "stream": true expected in body_json; cb receives each text delta. */
pankh_response pankh_chat_stream(const char *base_url, const char *body_json, const char **extra_headers, pankh_delta_cb cb, void *user);

/* POST to /v1/route: the decision only, no model call. */
pankh_response pankh_route(const char *base_url, const char *body_json, const char **extra_headers);

/* Build a minimal body: {"model":"auto","messages":[{"role":"user","content":question}],
 * "pankhllm":{...}} with optional context_json (a JSON array of chunks) and tags_json
 * (a JSON array of strings). Returns malloc'd string; free with pankh_free. */
char *pankh_simple_body(const char *question, const char *context_json, const char *tags_json, int stream);

void pankh_free(void *p);
#endif
