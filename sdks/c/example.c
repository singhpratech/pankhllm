#include "pankhllm.h"
#include <stdio.h>

static void on_delta(const char *d, size_t n, void *u) { (void)u; fwrite(d, 1, n, stdout); fflush(stdout); }

int main(void) {
    const char *headers[] = { "X-Pankh-Tags: private", NULL };
    char *body = pankh_simple_body("What was call activity for East in August?",
        "[{\"text\":\"East Aug 2026: 1,842 HCP calls, reach 71%.\",\"score\":0.9}]", NULL, 0);
    pankh_response r = pankh_chat("http://localhost:4000", body, headers);
    printf("status %ld\n%s\n", r.status, r.body);
    pankh_free(r.body); pankh_free(body);

    body = pankh_simple_body("Define the reach KPI in one sentence.", NULL, NULL, 1);
    r = pankh_chat_stream("http://localhost:4000", body, NULL, on_delta, NULL);
    printf("\n[stream status %ld]\n", r.status);
    pankh_free(r.body); pankh_free(body);
    return 0;
}
