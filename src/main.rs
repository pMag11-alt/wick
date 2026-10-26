//! The `wick` command line.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use wick::chat::Message;
use wick::engine::{Engine, GenOptions, Session};
use wick::model::{Format, State};
use wick::sampler::log_prob;

const USAGE: &str = "\
wick: run Llama and Qwen language models on the CPU

usage:
  wick run      <model-dir> -p <prompt>   generate a reply (chat format) or raw text (--raw)
  wick chat     <model-dir>               interactive chat
  wick serve    <model-dir>               OpenAI-compatible API and web UI (default 127.0.0.1:8080)
  wick bench    <model-dir>               prompt and generation speed
  wick ppl      <model-dir> <text-file>   perplexity of a text file
  wick tokenize <model-dir> <text>        show token ids

options:
  -p, --prompt <text>      prompt (run)
  -s, --system <text>      system message
  -n, --max-tokens <n>     tokens to generate (default 256)
  -t, --temperature <x>    sampling temperature, 0 = greedy (default 0.7)
      --top-k <n>          (default 40)     --top-p <x>   (default 0.9)
      --min-p <x>          (default 0)      --repeat-penalty <x> (default 1.0)
      --seed <n>           RNG seed
  -q, --quant <fmt>        weights in memory: q8 (default), q4, bf16 (exact), f32
  -j, --threads <n>        worker threads (default: cores, up to 8)
  -c, --context <n>        context length (default 4096)
      --raw                plain completion, no chat template
      --think / --no-think Qwen3 thinking mode
      --host <addr>        serve: listen address (default 127.0.0.1:8080)
      --ctx <n>            ppl: window length (default 512)
      --windows <n>        ppl: only the first n windows
      --prompt-len <n>     bench: prompt tokens (default 128)
";

struct Args {
    cmd: String,
    dir: PathBuf,
    rest: Vec<String>,
    prompt: Option<String>,
    system: Option<String>,
    opts: GenOptions,
    format: Format,
    threads: usize,
    context: usize,
    raw: bool,
    think: Option<bool>,
    host: String,
    ppl_ctx: usize,
    ppl_windows: usize,
    prompt_len: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or(USAGE)?;
    if cmd == "-h" || cmd == "--help" || cmd == "help" {
        return Err(USAGE.into());
    }
    let mut a = Args {
        cmd,
        dir: PathBuf::new(),
        rest: vec![],
        prompt: None,
        system: None,
        opts: GenOptions::default(),
        format: Format::Q8,
        threads: Engine::default_threads(),
        context: 4096,
        raw: false,
        think: None,
        host: "127.0.0.1:8080".into(),
        ppl_ctx: 512,
        ppl_windows: 0,
        prompt_len: 128,
    };
    let mut positional = vec![];
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        fn num<T: std::str::FromStr>(s: String, name: &str) -> Result<T, String> {
            s.parse().map_err(|_| format!("bad value for {name}: {s}"))
        }
        match arg.as_str() {
            "-p" | "--prompt" => a.prompt = Some(val(&arg)?),
            "-s" | "--system" => a.system = Some(val(&arg)?),
            "-n" | "--max-tokens" => a.opts.max_tokens = num(val(&arg)?, &arg)?,
            "-t" | "--temperature" => a.opts.sampler.temperature = num(val(&arg)?, &arg)?,
            "--top-k" => a.opts.sampler.top_k = num(val(&arg)?, &arg)?,
            "--top-p" => a.opts.sampler.top_p = num(val(&arg)?, &arg)?,
            "--min-p" => a.opts.sampler.min_p = num(val(&arg)?, &arg)?,
            "--repeat-penalty" => a.opts.sampler.repetition_penalty = num(val(&arg)?, &arg)?,
            "--seed" => a.opts.sampler.seed = num(val(&arg)?, &arg)?,
            "-q" | "--quant" => {
                let v = val(&arg)?;
                a.format = Format::parse(&v).ok_or_else(|| format!("unknown format {v} (bf16, f32, q8, q4)"))?;
            }
            "-j" | "--threads" => a.threads = num(val(&arg)?, &arg)?,
            "-c" | "--context" => a.context = num(val(&arg)?, &arg)?,
            "--raw" => a.raw = true,
            "--think" => a.think = Some(true),
            "--no-think" => a.think = Some(false),
            "--host" => a.host = val(&arg)?,
            "--ctx" => a.ppl_ctx = num(val(&arg)?, &arg)?,
            "--windows" => a.ppl_windows = num(val(&arg)?, &arg)?,
            "--prompt-len" => a.prompt_len = num(val(&arg)?, &arg)?,
            s if s.starts_with('-') && s.len() > 1 => return Err(format!("unknown option {s}\n\n{USAGE}")),
            _ => positional.push(arg),
        }
    }
    let mut pos = positional.into_iter();
    a.dir = pos.next().map(PathBuf::from).ok_or_else(|| format!("missing <model-dir>\n\n{USAGE}"))?;
    a.rest = pos.collect();
    Ok(a)
}

fn load(a: &Args) -> Result<Engine, String> {
    let mut e = Engine::load(&a.dir, a.format, a.threads)?;
    if a.think.is_some() && e.chat.thinking.is_some() {
        e.chat.thinking = a.think;
    }
    let m = &e.model;
    eprintln!(
        "wick: {} ({:.0}M params, {} layers, {}) as {}, {:.1} GiB, {} threads, {} kernels, loaded in {:.0} ms",
        e.name,
        m.param_count() as f64 / 1e6,
        m.cfg.layers,
        format!("{:?}", m.cfg.arch).to_lowercase(),
        m.format.name(),
        m.weight_bytes() as f64 / (1u64 << 30) as f64,
        e.pool.threads(),
        wick::model::kernel_name(),
        e.load_ms
    );
    Ok(e)
}

fn prompt_tokens(e: &Engine, a: &Args, history: &[Message]) -> Vec<u32> {
    if a.raw {
        let text: String = history.iter().map(|m| m.content.as_str()).collect();
        return e.tokenizer.encode(&text, true);
    }
    e.tokenizer.encode(&e.chat.render(history), false)
}

fn stats(r: &wick::engine::GenResult) {
    eprintln!(
        "\n[{} prompt tokens ({} cached) at {:.1} tok/s, {} generated at {:.1} tok/s, {:?}]",
        r.prompt_tokens,
        r.reused_tokens,
        r.prompt_tps(),
        r.tokens.len(),
        r.gen_tps(),
        r.finish
    );
}

fn cmd_run(a: &Args) -> Result<(), String> {
    let e = load(a)?;
    let prompt = a.prompt.clone().or_else(|| a.rest.first().cloned()).ok_or("run needs -p <prompt>")?;
    let mut msgs = vec![];
    if let Some(s) = &a.system {
        msgs.push(Message::new("system", s));
    }
    msgs.push(Message::new("user", &prompt));
    let ids = prompt_tokens(&e, a, &msgs);
    let mut session = Session::new(&e, a.context);
    if a.raw {
        print!("{prompt}");
    }
    let r = session.generate(&ids, &a.opts, |t| {
        print!("{t}");
        std::io::stdout().flush().ok();
        true
    })?;
    println!();
    stats(&r);
    Ok(())
}

fn cmd_chat(a: &Args) -> Result<(), String> {
    let e = load(a)?;
    let mut session = Session::new(&e, a.context);
    let mut history = vec![];
    if let Some(s) = &a.system {
        history.push(Message::new("system", s));
    }
    eprintln!("type a message, /reset to start over, Ctrl-D to quit");
    let stdin = std::io::stdin();
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/reset" {
            history.retain(|m: &Message| m.role == "system");
            continue;
        }
        history.push(Message::new("user", line));
        let ids = e.tokenizer.encode(&e.chat.render(&history), false);
        let r = session.generate(&ids, &a.opts, |t| {
            print!("{t}");
            std::io::stdout().flush().ok();
            true
        })?;
        println!();
        stats(&r);
        history.push(Message::new("assistant", &r.text));
    }
    Ok(())
}

fn cmd_bench(a: &Args) -> Result<(), String> {
    let e = load(a)?;
    let n = a.prompt_len;
    let n_gen = a.opts.max_tokens.min(128);
    // A deterministic, ordinary-looking prompt.
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(n);
    let mut ids = e.tokenizer.encode(&text, true);
    ids.truncate(n);
    let mut state = State::new(&e.model, n + n_gen + 8);
    // Warm up the page cache and the threads.
    e.model.forward(&e.pool, &mut state, &ids[..8.min(ids.len())], false);
    state.cache.len = 0;
    let t0 = Instant::now();
    let mut logits = Vec::new();
    for chunk in ids.chunks(256) {
        logits = e.model.forward(&e.pool, &mut state, chunk, false);
    }
    let pp = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut tok = wick::sampler::argmax(&logits);
    for _ in 0..n_gen {
        logits = e.model.forward(&e.pool, &mut state, &[tok], false);
        tok = wick::sampler::argmax(&logits);
    }
    let tg = t1.elapsed().as_secs_f64();
    println!(
        "{} {}: prompt {} tokens {:.1} tok/s | generate {} tokens {:.1} tok/s ({:.1} ms/token)",
        e.name,
        e.model.format.name(),
        ids.len(),
        ids.len() as f64 / pp,
        n_gen,
        n_gen as f64 / tg,
        tg * 1e3 / n_gen as f64
    );
    Ok(())
}

fn cmd_ppl(a: &Args) -> Result<(), String> {
    let e = load(a)?;
    let path = a.rest.first().ok_or("ppl needs a text file")?;
    let text = std::fs::read_to_string(path).map_err(|err| format!("{path}: {err}"))?;
    let ids = e.tokenizer.encode(&text, false);
    let ctx = a.ppl_ctx;
    // Non-overlapping windows; each predicts tokens 1..ctx from scratch.
    let mut windows = ids.len() / ctx;
    if a.ppl_windows > 0 {
        windows = windows.min(a.ppl_windows);
    }
    if windows == 0 {
        return Err("text is shorter than one window".into());
    }
    let mut state = State::new(&e.model, ctx);
    let (mut nll, mut count) = (0f64, 0usize);
    let t0 = Instant::now();
    for w in 0..windows {
        let win = &ids[w * ctx..(w + 1) * ctx];
        state.cache.len = 0;
        let vocab = e.model.cfg.vocab;
        for (c, chunk) in win.chunks(256).enumerate() {
            let logits = e.model.forward(&e.pool, &mut state, chunk, true);
            for (i, row) in logits.chunks_exact(vocab).enumerate() {
                let pos = c * 256 + i;
                if pos + 1 < ctx {
                    nll -= log_prob(row, win[pos + 1]);
                    count += 1;
                }
            }
        }
        eprint!("\r[{}/{}] ppl {:.4}", w + 1, windows, (nll / count as f64).exp());
    }
    eprintln!();
    println!(
        "{} {}: perplexity {:.4} over {} tokens ({} windows of {}), {:.1} tok/s",
        e.name,
        e.model.format.name(),
        (nll / count as f64).exp(),
        count,
        windows,
        ctx,
        (windows * ctx) as f64 / t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_tokenize(a: &Args) -> Result<(), String> {
    let tk = wick::tokenizer::Tokenizer::load_dir(&a.dir)?;
    let text = a.prompt.clone().or_else(|| a.rest.first().cloned()).ok_or("tokenize needs text")?;
    let ids = tk.encode(&text, false);
    for &id in &ids {
        println!("{id:>8}  {:?}", String::from_utf8_lossy(tk.token_bytes(id)));
    }
    eprintln!("{} tokens", ids.len());
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let result = match args.cmd.as_str() {
        "run" => cmd_run(&args),
        "chat" => cmd_chat(&args),
        "bench" => cmd_bench(&args),
        "ppl" => cmd_ppl(&args),
        "tokenize" => cmd_tokenize(&args),
        "serve" => load(&args).and_then(|e| wick::server::serve(e, &args.host, args.context)),
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("wick: {msg}");
            ExitCode::FAILURE
        }
    }
}
