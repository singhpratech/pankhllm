# pankhllm-client

Rust client for [pankhllm](https://github.com/singhpratech/pankhllm), the LLM gateway that
learns to skip the LLM. Async, built on reqwest.

```rust
use pankhllm_client::{Client, Options};

let pk = Client::new("http://localhost:4000");
let a = pk.ask("who covers R-3?", &Options { tags: vec!["private".into()], ..Default::default() }).await?;
println!("{} via {}", a.text, a.model);   // a.model == "decision:coverage_owner" when no LLM was called
```

The router itself: `cargo install pankhllm`, then `pankhllm serve --port 4000`.
Any OpenAI-compatible client works too; this one adds the `pankhllm` extras and typed results.
