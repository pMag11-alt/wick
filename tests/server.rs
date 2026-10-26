//! The HTTP server end to end, over a real socket. Needs SmolLM2-135M in
//! `$WICK_MODELS` (or `models/`); skipped without it.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use wick::engine::Engine;
use wick::json::Json;
use wick::model::Format;

/// A fresh server per test, so tests cannot evict each other's cache.
fn server() -> Option<SocketAddr> {
    {
        let root =
            std::env::var("WICK_MODELS").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models").into());
        let dir = PathBuf::from(root).join("SmolLM2-135M-Instruct");
        if !dir.join("model.safetensors").exists() {
            eprintln!("SKIPPED: {} not found", dir.display());
            return None;
        }
        let engine: &'static Engine = Box::leak(Box::new(Engine::load(&dir, Format::BF16, 4).unwrap()));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || wick::server::serve_listener(engine, listener, 2048));
        Some(addr)
    }
}

/// Send one HTTP request; return (status code, body).
fn http(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let status = resp[9..12].parse().unwrap();
    let body = resp.split_once("\r\n\r\n").unwrap().1.to_string();
    (status, body)
}

const CHAT: &str = r#"{"messages":[{"role":"user","content":"Name three primary colors."},{"role":"assistant","content":"Red, yellow and blue."},{"role":"user","content":"And the secondary ones?"}],"temperature":0,"max_tokens":24"#;

#[test]
fn chat_completion_streams_and_reuses_the_cache() {
    let Some(addr) = server() else { return };
    // Non-streaming: the same reply transformers gives (tests/fixtures/hf).
    let (status, body) = http(addr, "POST", "/v1/chat/completions", &format!("{CHAT}}}"));
    assert_eq!(status, 200, "{body}");
    let j = Json::parse(&body).unwrap();
    let reply = j.path(&["choices"]).unwrap().as_array().unwrap()[0]
        .path(&["message", "content"])
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(reply, "Green, orange, and purple.");
    assert_eq!(
        j.path(&["choices"]).unwrap().as_array().unwrap()[0].get("finish_reason").unwrap().as_str(),
        Some("stop")
    );

    // Streaming: SSE chunks that add up to the same reply, then [DONE]. The
    // identical prompt is served from the KV cache.
    let (status, body) = http(addr, "POST", "/v1/chat/completions", &format!("{CHAT},\"stream\":true}}"));
    assert_eq!(status, 200);
    let mut text = String::new();
    let mut cached = 0;
    let events: Vec<&str> = body.split("\n\n").filter(|e| e.starts_with("data: ")).collect();
    assert_eq!(events.last(), Some(&"data: [DONE]"));
    for e in &events[..events.len() - 1] {
        let j = Json::parse(&e[6..]).unwrap();
        let choice = &j.get("choices").unwrap().as_array().unwrap()[0];
        if let Some(c) = choice.path(&["delta", "content"]).and_then(Json::as_str) {
            text += c;
        }
        if let Some(n) = j.path(&["usage", "prompt_tokens_details", "cached_tokens"]).and_then(Json::as_usize) {
            cached = n;
        }
    }
    assert_eq!(text, reply);
    assert!(cached > 40, "expected the prompt to come from the cache, got {cached} cached tokens");
}

#[test]
fn completions_models_and_errors() {
    let Some(addr) = server() else { return };
    let (status, body) = http(addr, "GET", "/v1/models", "");
    assert_eq!(status, 200);
    assert!(body.contains("SmolLM2-135M-Instruct"));

    let (status, body) = http(
        addr,
        "POST",
        "/v1/completions",
        r#"{"prompt":"1, 2, 3, 4,","max_tokens":6,"temperature":0,"stop":["7"]}"#,
    );
    assert_eq!(status, 200, "{body}");
    let text = Json::parse(&body).unwrap().get("choices").unwrap().as_array().unwrap()[0]
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    assert!(text.starts_with(" 5, 6,"), "{text:?}");
    assert!(!text.contains('7'), "stop sequence leaked: {text:?}");

    for (path, body, want) in [
        ("/v1/chat/completions", "{not json", 400),
        ("/v1/chat/completions", r#"{"messages":[]}"#, 400),
        ("/v1/chat/completions", r#"{"messages":[{"role":"wizard","content":"hi"}]}"#, 400),
        ("/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}],"temperature":99}"#, 400),
        ("/nope", "", 404),
    ] {
        let (status, resp) = http(addr, "POST", path, body);
        assert_eq!(status, want, "{path} {body}: {resp}");
        if status == 400 {
            assert!(Json::parse(&resp).unwrap().path(&["error", "message"]).is_some());
        }
    }
    let (status, body) = http(addr, "GET", "/", "");
    assert_eq!(status, 200);
    assert!(body.contains("<title>wick</title>"));
}
