//! Chat prompt formats.
//!
//! Hugging Face models ship their prompt format as a Jinja template. Rather
//! than embed a template engine, wick recognises the format family from the
//! special tokens the tokenizer defines and renders it directly, matching
//! what `apply_chat_template(..., add_generation_prompt=True)` produces.

use crate::tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Template {
    /// `<|im_start|>role\n...<|im_end|>\n` (Qwen, SmolLM)
    ChatMl,
    /// `<|start_header_id|>role<|end_header_id|>\n\n...<|eot_id|>` (Llama 3)
    Llama3,
    /// `<|user|>\n...</s>\n` (Zephyr, TinyLlama)
    Zephyr,
    /// No chat format: messages joined as plain text.
    Plain,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Message {
        Message { role: role.to_string(), content: content.to_string() }
    }
}

#[derive(Clone, Debug)]
pub struct ChatFormat {
    pub template: Template,
    /// Used when the conversation has no system message (as the model's own template does).
    pub default_system: Option<String>,
    /// Qwen3: whether the model should think before answering.
    pub thinking: Option<bool>,
}

impl ChatFormat {
    /// Pick the format for a model from its tokenizer and config.json architecture.
    pub fn detect(tk: &Tokenizer, model_dir: &std::path::Path) -> ChatFormat {
        let has = |t: &str| tk.token_id(t).is_some();
        let template = if has("<|im_start|>") && has("<|im_end|>") {
            Template::ChatMl
        } else if has("<|start_header_id|>") && has("<|eot_id|>") {
            Template::Llama3
        } else if has("<|user|>") || tk.token_id("</s>").is_some() {
            Template::Zephyr
        } else {
            Template::Plain
        };
        // The default system prompts below are what each family's template inserts.
        let tc = std::fs::read_to_string(model_dir.join("tokenizer_config.json")).unwrap_or_default();
        let default_system = if tc.contains("You are Qwen, created by Alibaba Cloud") {
            Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".to_string())
        } else if tc.contains("named SmolLM, trained by Hugging Face") {
            Some("You are a helpful AI assistant named SmolLM, trained by Hugging Face".to_string())
        } else {
            None
        };
        let thinking = if tc.contains("enable_thinking") { Some(true) } else { None };
        ChatFormat { template, default_system, thinking }
    }

    /// Render messages followed by the prompt for the assistant's reply.
    pub fn render(&self, messages: &[Message]) -> String {
        let mut msgs: Vec<Message> = messages.to_vec();
        if msgs.first().is_none_or(|m| m.role != "system")
            && let Some(sys) = &self.default_system
        {
            msgs.insert(0, Message::new("system", sys));
        }
        let mut out = String::new();
        match self.template {
            Template::ChatMl => {
                for m in &msgs {
                    let content = if m.role == "assistant" && self.thinking.is_some() {
                        strip_thinking(&m.content)
                    } else {
                        &m.content
                    };
                    out += &format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, content);
                }
                out += "<|im_start|>assistant\n";
                if self.thinking == Some(false) {
                    out += "<think>\n\n</think>\n\n";
                }
            }
            Template::Llama3 => {
                out += "<|begin_of_text|>";
                for m in &msgs {
                    out += &format!("<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>", m.role, m.content.trim());
                }
                out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
            }
            Template::Zephyr => {
                for m in &msgs {
                    out += &format!("<|{}|>\n{}</s>\n", m.role, m.content);
                }
                out += "<|assistant|>\n";
            }
            Template::Plain => {
                for m in &msgs {
                    out += &m.content;
                    out += "\n";
                }
            }
        }
        out
    }
}

/// Qwen3 keeps only the final answer of earlier assistant turns.
fn strip_thinking(s: &str) -> &str {
    match s.rfind("</think>") {
        Some(i) => s[i + "</think>".len()..].trim_start_matches('\n'),
        None => s,
    }
}
