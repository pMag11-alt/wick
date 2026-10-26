//! An OpenAI-compatible HTTP server on `std::net`.
//!
//!   GET  /                      a small chat web UI
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/chat/completions   messages -> reply, optionally streamed (SSE)
//!   POST /v1/completions        raw text completion
//!
//! Requests share one KV cache. A chat that grows turn by turn resends the
//! whole conversation each time, and only the new tokens are processed.
//! Requests run one at a time; a client that disconnects mid-stream
//! cancels its generation.

use crate::chat::Message;
use crate::engine::{Engine, Finish, GenOptions, GenResult, Session};
use crate::json::Json;
use crate::sampler::SamplerConfig;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const UI: &str = include_str!("ui.html");
const MAX_BODY: usize = 8 << 20;

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut BufReader<TcpStream>) -> Result<Option<Request>, String> {
    let mut line = String::new();
    if stream.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("bad request line")?.to_string();
    let path = parts.next().ok_or("bad request line")?.split('?').next().unwrap().to_string();
    let mut length = 0usize;
    let mut header_bytes = 0;
    loop {
        let mut h = String::new();
        stream.read_line(&mut h).map_err(|e| e.to_string())?;
        header_bytes += h.len();
        if header_bytes > 64 << 10 {
            return Err("headers too large".into());
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            if k == "content-length" {
                length = v.trim().parse().map_err(|_| "bad Content-Length")?;
            } else if k == "transfer-encoding" {
                return Err("chunked request bodies are not supported".into());
            }
        }
    }
    if length > MAX_BODY {
        return Err("request body too large".into());
    }
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Some(Request { method, path, body }))
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn json_response(stream: &mut TcpStream, status: &str, j: &Json) -> std::io::Result<()> {
    respond(stream, status, "application/json", j.to_string().as_bytes())
}

fn error_json(message: &str, kind: &str) -> Json {
    Json::object([("error", Json::object([("message", Json::str(message)), ("type", Json::str(kind))]))])
}

fn now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as f64
}

/// Text of a message's content: a string, or an array of {type: text} parts.
fn content_text(c: &Json) -> Result<String, String> {
    match c {
        Json::String(s) => Ok(s.clone()),
        Json::Array(parts) => Ok(parts
            .iter()
            .filter(|p| p.get("type").and_then(Json::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Json::as_str))
            .collect::<Vec<_>>()
            .join("")),
        Json::Null => Ok(String::new()),
        _ => Err("message content must be a string or an array of parts".into()),
    }
}

fn gen_options(req: &Json) -> Result<GenOptions, String> {
    let mut o = GenOptions { max_tokens: 512, sampler: SamplerConfig::default(), stop_strings: vec![] };
    let f = |k: &str| req.get(k).and_then(Json::as_f64);
    if let Some(n) = req.get("max_completion_tokens").or_else(|| req.get("max_tokens")).and_then(Json::as_usize) {
        o.max_tokens = n;
    }
    if let Some(t) = f("temperature") {
        o.sampler.temperature = t as f32;
    }
    if let Some(p) = f("top_p") {
        o.sampler.top_p = p as f32;
    }
    if let Some(k) = req.get("top_k").and_then(Json::as_usize) {
        o.sampler.top_k = k;
    }
    if let Some(p) = f("min_p") {
        o.sampler.min_p = p as f32;
    }
    if let Some(p) = f("repetition_penalty") {
        o.sampler.repetition_penalty = p as f32;
    }
    o.sampler.seed = req.get("seed").and_then(Json::as_i64).map(|s| s as u64).unwrap_or_else(|| now() as u64 ^ 0x9E37);
    match req.get("stop") {
        Some(Json::String(s)) => o.stop_strings.push(s.clone()),
        Some(Json::Array(a)) => o.stop_strings.extend(a.iter().filter_map(Json::as_str).map(String::from)),
        _ => {}
    }
    o.stop_strings.retain(|s| !s.is_empty());
    if !(0.0..=10.0).contains(&o.sampler.temperature) || !(0.0..=1.0).contains(&o.sampler.top_p) {
        return Err("temperature must be in [0, 10] and top_p in [0, 1]".into());
    }
    Ok(o)
}

struct Server<'e> {
    engine: &'e Engine,
    session: Mutex<Session<'e>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl Server<'_> {
    fn completion(&self, stream: &mut TcpStream, req: &Json, chat: bool) -> Result<(), String> {
        let e = self.engine;
        let opts = gen_options(req)?;
        let prompt_ids = if chat {
            let msgs = req.get("messages").and_then(Json::as_array).ok_or("messages is required")?;
            let mut messages = Vec::new();
            for m in msgs {
                let role = m.get("role").and_then(Json::as_str).ok_or("each message needs a role")?;
                if !matches!(role, "system" | "user" | "assistant" | "developer") {
                    return Err(format!("unsupported role {role}"));
                }
                let role = if role == "developer" { "system" } else { role };
                messages.push(Message::new(role, &content_text(m.get("content").unwrap_or(&Json::Null))?));
            }
            if messages.is_empty() {
                return Err("messages is empty".into());
            }
            e.tokenizer.encode(&e.chat.render(&messages), false)
        } else {
            let prompt = req.get("prompt").and_then(Json::as_str).ok_or("prompt must be a string")?;
            e.tokenizer.encode(prompt, true)
        };
        let streaming = req.get("stream").and_then(Json::as_bool).unwrap_or(false);
        let id = format!(
            "{}-{}",
            if chat { "chatcmpl" } else { "cmpl" },
            self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let object = if chat { "chat.completion" } else { "text_completion" };
        let created = Json::num(now());
        let model = Json::str(&e.name);

        let mut session = self.session.lock().unwrap();
        if prompt_ids.len() >= session.context() {
            return Err(format!("prompt is {} tokens but the context is {}", prompt_ids.len(), session.context()));
        }

        let chunk = |delta: Json, finish: Option<&str>| -> Json {
            let choice = if chat {
                Json::object([
                    ("index", Json::num(0)),
                    ("delta", delta),
                    ("finish_reason", finish.map(Json::str).unwrap_or(Json::Null)),
                ])
            } else {
                let text = delta.get("content").cloned().unwrap_or(Json::str(""));
                Json::object([
                    ("index", Json::num(0)),
                    ("text", text),
                    ("finish_reason", finish.map(Json::str).unwrap_or(Json::Null)),
                ])
            };
            Json::object([
                ("id", Json::str(&id)),
                ("object", Json::str(if chat { "chat.completion.chunk" } else { "text_completion" })),
                ("created", created.clone()),
                ("model", model.clone()),
                ("choices", Json::Array(vec![choice])),
            ])
        };

        if streaming {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
                 Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
            )
            .map_err(|e| e.to_string())?;
            let send =
                |s: &mut TcpStream, j: &Json| -> bool { write!(s, "data: {j}\n\n").and_then(|_| s.flush()).is_ok() };
            if chat {
                send(
                    stream,
                    &chunk(Json::object([("role", Json::str("assistant")), ("content", Json::str(""))]), None),
                );
            }
            let r = session.generate(&prompt_ids, &opts, |text| {
                send(stream, &chunk(Json::object([("content", Json::str(text))]), None))
            })?;
            if r.finish != Finish::Cancelled {
                let mut last = chunk(Json::Object(vec![]), Some(finish_reason(&r)));
                if let Json::Object(fields) = &mut last {
                    fields.push(("usage".into(), usage(&r)));
                    fields.push(("timings".into(), timings(&r)));
                }
                send(stream, &last);
                let _ = write!(stream, "data: [DONE]\n\n");
            }
            log_request(&id, &r);
            return Ok(());
        }

        let r = session.generate(&prompt_ids, &opts, |_| true)?;
        drop(session);
        log_request(&id, &r);
        let choice = if chat {
            Json::object([
                ("index", Json::num(0)),
                ("message", Json::object([("role", Json::str("assistant")), ("content", Json::str(&r.text))])),
                ("finish_reason", Json::str(finish_reason(&r))),
            ])
        } else {
            Json::object([
                ("index", Json::num(0)),
                ("text", Json::str(&r.text)),
                ("finish_reason", Json::str(finish_reason(&r))),
            ])
        };
        let body = Json::object([
            ("id", Json::str(&id)),
            ("object", Json::str(object)),
            ("created", created),
            ("model", model),
            ("choices", Json::Array(vec![choice])),
            ("usage", usage(&r)),
            ("timings", timings(&r)),
        ]);
        json_response(stream, "200 OK", &body).map_err(|e| e.to_string())
    }

    fn handle(&self, stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().expect("clone socket"));
        let mut stream = stream;
        let req = match read_request(&mut reader) {
            Ok(Some(r)) => r,
            Ok(None) => return,
            Err(msg) => {
                let _ = json_response(&mut stream, "400 Bad Request", &error_json(&msg, "invalid_request_error"));
                return;
            }
        };
        let result = match (req.method.as_str(), req.path.as_str()) {
            ("OPTIONS", _) => respond(&mut stream, "204 No Content", "text/plain", b"").map_err(|e| e.to_string()),
            ("GET", "/") => {
                respond(&mut stream, "200 OK", "text/html; charset=utf-8", UI.as_bytes()).map_err(|e| e.to_string())
            }
            ("GET", "/health") => json_response(&mut stream, "200 OK", &Json::object([("status", Json::str("ok"))]))
                .map_err(|e| e.to_string()),
            ("GET", "/v1/models") => {
                let m = Json::object([
                    ("id", Json::str(&self.engine.name)),
                    ("object", Json::str("model")),
                    ("owned_by", Json::str("wick")),
                    ("quantization", Json::str(self.engine.model.format.name())),
                    ("context_length", Json::num(self.session.lock().unwrap().context() as f64)),
                ]);
                json_response(
                    &mut stream,
                    "200 OK",
                    &Json::object([("object", Json::str("list")), ("data", Json::Array(vec![m]))]),
                )
                .map_err(|e| e.to_string())
            }
            ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
                let chat = req.path == "/v1/chat/completions";
                match std::str::from_utf8(&req.body)
                    .map_err(|_| "body is not UTF-8".to_string())
                    .and_then(|t| Json::parse(t).map_err(|e| e.to_string()))
                {
                    Ok(body) => self.completion(&mut stream, &body, chat).or_else(|msg| {
                        json_response(&mut stream, "400 Bad Request", &error_json(&msg, "invalid_request_error"))
                            .map_err(|e| e.to_string())
                    }),
                    Err(msg) => {
                        json_response(&mut stream, "400 Bad Request", &error_json(&msg, "invalid_request_error"))
                            .map_err(|e| e.to_string())
                    }
                }
            }
            (_, "/v1/chat/completions" | "/v1/completions" | "/v1/models" | "/health" | "/") => json_response(
                &mut stream,
                "405 Method Not Allowed",
                &error_json("method not allowed", "invalid_request_error"),
            )
            .map_err(|e| e.to_string()),
            _ => json_response(&mut stream, "404 Not Found", &error_json("not found", "invalid_request_error"))
                .map_err(|e| e.to_string()),
        };
        if let Err(e) = result {
            eprintln!("wick: request failed: {e}");
        }
    }
}

fn finish_reason(r: &GenResult) -> &'static str {
    match r.finish {
        Finish::Length => "length",
        _ => "stop",
    }
}

fn usage(r: &GenResult) -> Json {
    Json::object([
        ("prompt_tokens", Json::num(r.prompt_tokens as f64)),
        ("completion_tokens", Json::num(r.tokens.len() as f64)),
        ("total_tokens", Json::num((r.prompt_tokens + r.tokens.len()) as f64)),
        ("prompt_tokens_details", Json::object([("cached_tokens", Json::num(r.reused_tokens as f64))])),
    ])
}

fn timings(r: &GenResult) -> Json {
    Json::object([
        ("prompt_ms", Json::num((r.prompt_ms * 10.0).round() / 10.0)),
        ("prompt_per_second", Json::num((r.prompt_tps() * 10.0).round() / 10.0)),
        ("predicted_ms", Json::num((r.gen_ms * 10.0).round() / 10.0)),
        ("predicted_per_second", Json::num((r.gen_tps() * 10.0).round() / 10.0)),
    ])
}

fn log_request(id: &str, r: &GenResult) {
    eprintln!(
        "wick: {id}: {} prompt tokens ({} cached) {:.0} tok/s, {} generated {:.1} tok/s, {:?}",
        r.prompt_tokens,
        r.reused_tokens,
        r.prompt_tps(),
        r.tokens.len(),
        r.gen_tps(),
        r.finish
    );
}

pub fn serve(engine: Engine, addr: &str, context: usize) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
    eprintln!("wick: serving {} on http://{addr} (OpenAI API at /v1, chat UI at /)", engine.name);
    serve_listener(&engine, listener, context);
    Ok(())
}

/// Serve requests on an already bound listener (tests bind port 0).
pub fn serve_listener(engine: &Engine, listener: TcpListener, context: usize) {
    let server = Server { engine, session: Mutex::new(Session::new(engine, context)), next_id: Default::default() };
    std::thread::scope(|scope| {
        for conn in listener.incoming() {
            match conn {
                Ok(stream) => {
                    let server = &server;
                    scope.spawn(move || server.handle(stream));
                }
                Err(e) => eprintln!("wick: accept failed: {e}"),
            }
        }
    });
}
