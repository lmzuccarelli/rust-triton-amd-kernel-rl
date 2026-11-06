use crate::config::load::LlmAgent;
use crate::inference::llm::{
    LlmClaude, LlmInterface, LlmInterfaceOpenApi, LlmOpenApi, LlmOpenCode,
};
use crate::utils::common::get_item;
use custom_logger as log;
use http::{Method, Request, Response, StatusCode};
use http_body_util::BodyExt;
use http_body_util::Full;
use hyper::body::{Bytes, Incoming};
use std::str::FromStr;

pub async fn endpoints(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let mut response = Response::new(Full::default());
    let parameters = req.uri().path().split("/").collect::<Vec<&str>>();
    let agent_type = get_agent(parameters.clone());
    log::debug!("[endpoints] llm request {:?}", parameters);
    match *req.method() {
        Method::POST => {
            let data = req.into_body().collect().await?.to_bytes();
            let prompt_res = String::from_utf8(data.to_vec());
            match prompt_res {
                Ok(prompt) => {
                    let (status, result) = execute_agent(agent_type, prompt).await;
                    match status {
                        StatusCode::OK => {
                            *response.body_mut() = Full::from(result);
                        }
                        _ => {
                            log::error!("{}", result);
                            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                            *response.body_mut() = Full::from(result);
                        }
                    }
                }
                Err(err) => {
                    log::error!("[endpoints] prompt request parsing body error {}", err);
                    *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                    *response.body_mut() = Full::from(format!(
                        "[endpoints] prompt request parsing body error {}\n",
                        err
                    ));
                }
            }
        }
        Method::GET => {
            let mut content = format!(
                r##"{{ "status": "ok", "appplication": "{}", "service": "llm-inference", "version": "{}" }}"##,
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            );
            content.push('\n');
            *response.body_mut() = Full::from(content);
        }
        _ => {
            log::error!("[endpoints] method/endpoint not implemented");
            *response.body_mut() = Full::from("[endpoints] method/endpoint not implmented\n");
            *response.status_mut() = StatusCode::NOT_FOUND;
        }
    };
    Ok(response)
}

// helper functions

// execute the agent depending on type
// handle the error with StatusCode
async fn execute_agent(agent_type: LlmAgent, prompt: String) -> (StatusCode, String) {
    log::debug!("[execute_agent] agent type {:?}", agent_type);
    let (status, result) = match agent_type {
        LlmAgent::Claude => {
            let res = LlmClaude::run(prompt).await;
            match res {
                Ok(contents) => (StatusCode::OK, contents),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("[execute_agent] claude execution failed {}", e),
                ),
            }
        }
        LlmAgent::Opencode => {
            let res = LlmOpenCode::run(prompt).await;
            match res {
                Ok(contents) => (StatusCode::OK, contents),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("[execute_agent] opencode execution failed {}", e),
                ),
            }
        }
        LlmAgent::Api => {
            let token = get_item("token").unwrap_or("".to_string());
            let url = get_item("openapi_url").unwrap_or("".to_string());
            let model = get_item("model").unwrap_or("".to_string());
            let response_type = get_item("response_type").unwrap_or("".to_string());
            let res = LlmOpenApi::run(prompt, url, token, model, response_type).await;
            match res {
                Ok(contents) => (StatusCode::OK, contents),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("[execute_agent] openapi execution failed {}", e),
                ),
            }
        }
    };
    (status, result)
}

fn get_agent(params: Vec<&str>) -> LlmAgent {
    LlmAgent::from_str(params.last().unwrap_or(&"none")).unwrap_or(LlmAgent::Claude)
}
