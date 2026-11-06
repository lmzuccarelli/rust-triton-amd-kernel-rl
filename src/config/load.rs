use serde_derive::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::str::FromStr;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Parameters {
    pub name: String,
    pub log_level: String,
    pub gpu_server_url: String,
    pub llm_server_url: String,
    pub workflow_batch: Vec<String>,
    pub working_dir: String,
    pub llm_model: String,
    pub llm_agent: LlmAgent,
    pub cutedsl_env_path: String,
    pub controller_mode: ControllerMode,
    pub token_file: Option<String>,
    pub openapi_url: Option<String>,
    pub response_type: Option<ResponseType>,
    pub gpu_arch: u8,
    pub max_rollout: u8,
    pub rollout_start: u8,
    pub flow_control: u8,
    pub create_stats: bool,
    pub use_error_vec: bool,
    pub exclude_trajectories: Vec<String>,
    pub acceptance_threshold: f64,
    pub degradation_threshold: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorkItem {
    pub name: String,
    pub gpu_arch: String,
    pub prompt: Option<String>,
    pub target_dir: String,
    pub working_dir: String,
    pub kernel_file: Option<String>,
    pub code: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum LlmAgent {
    Claude,
    Opencode,
    Api,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ControllerMode {
    Baseline,
    Agent,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ResponseType {
    Enmaas,
    Openai,
}

impl fmt::Display for LlmAgent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for LlmAgent {
    type Err = ();
    fn from_str(input: &str) -> Result<LlmAgent, Self::Err> {
        match input {
            "claude" => Ok(LlmAgent::Claude),
            "opencode" => Ok(LlmAgent::Opencode),
            "api" => Ok(LlmAgent::Api),
            _ => Err(()),
        }
    }
}

impl fmt::Display for ResponseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait ConfigInterface {
    fn read(&self, dir: String) -> Result<Parameters, Box<dyn std::error::Error>>;
}

#[derive(Debug, Clone)]
pub struct ImplConfigInterface {}

impl ConfigInterface for ImplConfigInterface {
    fn read(&self, name: String) -> Result<Parameters, Box<dyn std::error::Error>> {
        let json_data = File::open(&name)?;
        let params = serde_json::from_reader(json_data)?;
        Ok(params)
    }
}
