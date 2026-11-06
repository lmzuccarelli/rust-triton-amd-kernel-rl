use custom_logger as log;
use std::process::Command;

pub trait ShellExecuteInterface {
    fn run(cmd: &str, args: Vec<&str>) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct ShellExecute {}

impl ShellExecuteInterface for ShellExecute {
    fn run(cmd: &str, args: Vec<&str>) -> Result<String, Box<dyn std::error::Error>> {
        log::debug!("[run] executing command {}", cmd);
        let output = Command::new(cmd)
            .args(args)
            .output()
            .expect("failed to spawn shell");

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(stdout)
    }
}
