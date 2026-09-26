# Multi-stage build: static-ish release binary on a slim runtime image.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/target/release/pankhllm /usr/local/bin/pankhllm
COPY pankhllm.yaml ./pankhllm.yaml
COPY prompts ./prompts
EXPOSE 4000
ENV RUST_LOG=info
HEALTHCHECK --interval=15s --timeout=3s CMD ["/usr/local/bin/pankhllm", "--config", "/app/pankhllm.yaml", "check"]
ENTRYPOINT ["pankhllm", "--config", "/app/pankhllm.yaml"]
CMD ["serve", "--port", "4000"]
