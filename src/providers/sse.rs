//! Minimal server-sent-events decoder over a reqwest byte stream.

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};

use super::ProviderError;

/// Yields the `data:` payload of each SSE event. Multi-line data is joined with `\n`.
pub fn data_lines<S>(bytes: S) -> impl Stream<Item = Result<String, ProviderError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buf = BytesMut::new();
    let mut pending: Vec<String> = Vec::new();

    bytes.flat_map(move |chunk| {
        let mut out: Vec<Result<String, ProviderError>> = Vec::new();
        match chunk {
            Err(e) => out.push(Err(e.into())),
            Ok(b) => {
                buf.extend_from_slice(&b);
                // Process every complete line; keep the tail.
                while let Some(pos) = buf.iter().position(|&c| c == b'\n') {
                    let line = buf.split_to(pos + 1);
                    let line = String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string();
                    if line.is_empty() {
                        if !pending.is_empty() {
                            out.push(Ok(pending.join("\n")));
                            pending.clear();
                        }
                    } else if let Some(data) = line.strip_prefix("data:") {
                        pending.push(data.trim_start().to_string());
                    }
                    // `event:` / `id:` / comments are ignored; payload types carry their own tag.
                }
            }
        }
        futures::stream::iter(out)
    })
}
