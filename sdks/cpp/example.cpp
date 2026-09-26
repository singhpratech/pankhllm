#include "pankhllm.hpp"
#include <iostream>

int main() {
    pankhllm::Client pk("http://localhost:4000", {"X-Pankh-Tags: private"});
    std::cout << pk.chat(pankhllm::Client::simple_body("What was call activity for East in August?",
        R"([{"text":"East Aug 2026: 1,842 HCP calls, reach 71%.","score":0.9}])")) << "\n";
    pk.chat_stream(pankhllm::Client::simple_body("Define the reach KPI in one sentence.", "", "", true),
        [](const std::string& d) { std::cout << d << std::flush; });
    std::cout << "\n";
}
