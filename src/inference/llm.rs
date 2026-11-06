use custom_logger as log;
use hyper::StatusCode;
use reqwest::Client;
use serde_derive::Deserialize;
use serde_derive::Serialize;
use serde_json::Value;
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

// openapi schema

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnMaasResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    #[serde(rename = "stop_reason")]
    pub stop_reason: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub usage: EnMaasUsage,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    pub signature: Option<String>,
    pub thinking: Option<String>,
    #[serde(rename = "type")]
    pub type_field: String,
    pub text: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnMaasUsage {
    #[serde(rename = "cache_creation_input_tokens")]
    pub cache_creation_input_tokens: i64,
    #[serde(rename = "cache_read_input_tokens")]
    pub cache_read_input_tokens: i64,
    #[serde(rename = "input_tokens")]
    pub input_tokens: i64,
    #[serde(rename = "output_tokens")]
    pub output_tokens: i64,
    #[serde(rename = "prompt_tokens_details")]
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenaiChatCompletions {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(rename = "service_tier")]
    pub service_tier: Option<String>,
    #[serde(rename = "system_fingerprint")]
    pub system_fingerprint: Value,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Choice {
    pub index: i64,
    pub message: Message,
    #[serde(rename = "finish_reason")]
    pub finish_reason: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: String,
    pub content: String,
    pub refusal: Option<Value>,
    pub annotations: Option<Vec<Value>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    #[serde(rename = "prompt_tokens")]
    pub prompt_tokens: Option<i64>,
    #[serde(rename = "completion_tokens")]
    pub completion_tokens: Option<i64>,
    #[serde(rename = "total_tokens")]
    pub total_tokens: Option<i64>,
    #[serde(rename = "prompt_tokens_details")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(rename = "completion_tokens_details")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTokensDetails {
    #[serde(rename = "cached_tokens")]
    pub cached_tokens: i64,
    #[serde(rename = "audio_tokens")]
    pub audio_tokens: Option<i64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionTokensDetails {
    #[serde(rename = "reasoning_tokens")]
    pub reasoning_tokens: i64,
    #[serde(rename = "audio_tokens")]
    pub audio_tokens: Option<i64>,
    #[serde(rename = "accepted_prediction_tokens")]
    pub accepted_prediction_tokens: i64,
    #[serde(rename = "rejected_prediction_tokens")]
    pub rejected_prediction_tokens: i64,
}

#[allow(unused)]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSchema {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    pub temperature: f64,
    pub top_p: f64,
    pub stream: bool,
    pub max_tokens: i32,
    pub chat_template_kwargs: Option<ChatTemplate>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatTemplate {
    enable_thinking: bool,
}

#[allow(unused)]
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestMessage {
    pub role: String,
    pub content: String,
}

pub trait LlmInterface {
    async fn run(prompt: String) -> Result<String, Box<dyn std::error::Error>>;
}

pub trait LlmInterfaceOpenApi {
    async fn run(
        prompt: String,
        url: String,
        token: String,
        model: String,
        response_type: String,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct LlmClaude {}
pub struct LlmOpenCode {}
pub struct LlmOpenApi {}

impl LlmInterface for LlmClaude {
    async fn run(prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm claude inference endpoint");
        let start = Instant::now();

        // TODO: address ARG_MAX issues
        let output = Command::new("claude").args(vec!["-p", &prompt]).output()?;

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        // preserve output
        println!("{}", stdout);
        log::info!("[run] executing llm claude completed task in {:?}", elapsed);

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        Ok(stdout)
    }
}

impl LlmInterface for LlmOpenCode {
    async fn run(prompt: String) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm opencode inference endpoint");
        let start = Instant::now();

        let output = Command::new("opencode")
            .args(vec!["-p", &prompt])
            .output()?;

        let elapsed = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        // preserve output
        println!("{}", stdout);
        log::info!(
            "[run] executing llm opencode completed task in {:?}",
            elapsed
        );

        if !output.status.success() {
            // return the first failure (fail fast)
            return Err(Box::from(stderr));
        }

        Ok(stdout)
    }
}

impl LlmInterfaceOpenApi for LlmOpenApi {
    async fn run(
        prompt: String,
        url: String,
        token: String,
        model: String,
        response_type: String,
    ) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing llm {} inference endpoint", response_type);
        log::debug!("[run] executing llm url {}", url);
        log::debug!("[run] executing llm model {}", model);
        log::debug!("[run] executing llm prompt len {}", prompt.len());
        log::debug!("[run] executing llm prompt len {}", prompt.len());

        let start = Instant::now();

        let system_msg = RequestMessage {
            role: "user".to_string(),
            content: "you are a cuda kernel expert, help the user with generating code, analyzing and profiling of cuda kernels".to_string(),
        };
        let user_msg = RequestMessage {
            role: "user".to_owned(),
            content: prompt,
        };
        let vec_msgs = vec![system_msg, user_msg];

        // TODO: check if kwargs is supported
        // This is used to suppress thinking in qwen3.8-27B
        let ct = ChatTemplate {
            enable_thinking: false,
        };

        let rs = RequestSchema {
            model,
            temperature: 0.1,
            stream: false,
            max_tokens: 32000,
            top_p: 0.8,
            messages: vec_msgs,
            chat_template_kwargs: Some(ct.clone()),
        };
        let json_request = serde_json::to_string(&rs)?;
        log::trace!("[run] openapi json payload {}", json_request);

        let client_res = Client::builder()
            .danger_accept_invalid_certs(true)
            .http1_title_case_headers()
            .timeout(Duration::new(1200, 0))
            .build();

        let client = match client_res {
            Ok(client) => {
                log::debug!("[run] llm openapi client created");
                client
            }
            Err(e) => {
                return Err(Box::from(format!("[run] llm openapi {} ", e)));
            }
        };

        let client_res = client
            .post(url)
            .bearer_auth(token.trim())
            .header("Content-Type", "application/json")
            .header("X-Api-Key", token.trim())
            .body(json_request)
            .send()
            .await;

        let response = match client_res {
            Ok(result) => {
                let status = result.status();
                log::debug!("[run] llm openapi response status {}", status);
                match status {
                    StatusCode::OK => {
                        let contents = result.bytes().await?;
                        log::trace!(
                            "[run] llm openapi client response {}",
                            String::from_utf8(contents.to_vec()).unwrap()
                        );
                        match response_type.as_str() {
                            "Enmaas" => {
                                let chat_response: EnMaasResponse =
                                    serde_json::from_slice(&contents)?;
                                chat_response.content[0].text.clone()
                            }
                            "Openai" => {
                                let chat_response: OpenaiChatCompletions =
                                    serde_json::from_slice(&contents)?;
                                chat_response.choices[0].message.content.clone()
                            }
                            &_ => {
                                let chat_response: OpenaiChatCompletions =
                                    serde_json::from_slice(&contents)?;
                                chat_response.choices[0].message.content.clone()
                            }
                        }
                    }
                    _ => {
                        let contents = result.bytes().await?;
                        return Err(Box::from(format!(
                            "[run] llm openapi {}",
                            String::from_utf8(contents.to_vec())
                                .unwrap_or("could not parse error".to_string())
                        )));
                    }
                }
            }
            Err(e) => {
                return Err(Box::from(format!("[run] llm openapi error {}", e)));
            }
        };

        let elapsed = start.elapsed();
        log::info!(
            "[run] executing llm openapi completed task in {:?}",
            elapsed
        );

        Ok(response)
    }
}
