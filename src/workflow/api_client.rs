use custom_logger as log;
use hyper::StatusCode;
use reqwest::Client;
use std::fs;
use std::time::Duration;

// this is a complex post as it will call the endpoint
pub async fn process_post_call(
    file_name: Option<String>,
    url: String,
    data: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(Duration::new(1500, 0))
        .build()?;
    let client_response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(data)
        .send()
        .await?;

    let status = client_response.status();
    let response = client_response.bytes().await?;
    let doc_content = String::from_utf8(response.to_vec())?;

    // write to file if file_name is set
    // regardless of pass/fail
    match file_name {
        Some(name) => {
            fs::write(name, doc_content.clone())?;
        }
        None => {
            log::trace!("[process_post_call] file_name not supplied (not saving to disk)");
        }
    }
    match status {
        StatusCode::OK => {
            log::trace!("[process_post_call] completed successfully");
        }
        _ => {
            return Err(Box::from(format!(
                "[process_post_call] failed with status {}",
                status
            )));
        }
    };
    Ok(doc_content)
}

pub async fn process_get_call(url: String) -> Result<String, Box<dyn std::error::Error>> {
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(Duration::new(1500, 0))
        .build()?;
    log::trace!("[process_get_call] {}", url);
    let client_response = client.get(url).send().await?;

    if client_response.status() != StatusCode::OK {
        return Err(Box::from(format!(
            "[process_get_call] error status code {}",
            client_response.status()
        )));
    }
    log::trace!("[process_get_call] status {}", client_response.status());
    let response = client_response.bytes().await?;
    let result = str::from_utf8(&response)?;
    Ok(result.to_string())
}
