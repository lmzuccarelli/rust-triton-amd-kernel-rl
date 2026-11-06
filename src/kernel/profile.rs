use crate::config::load::WorkItem;
use crate::utils::common::get_item;
use custom_logger as log;
use regex::Regex;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub trait ProfileInterface {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>>;
    fn get_elapsed_cycles(ncu_report: String) -> Result<i64, Box<dyn std::error::Error>>;
    fn calculate_improvement(
        baseline: i64,
        current: i64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>>;
    fn get_category(state_report: String) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Profile {}

impl ProfileInterface for Profile {
    async fn run(work_item: WorkItem) -> Result<String, Box<dyn std::error::Error>> {
        let start = Instant::now();
        let mut profile_buffer = String::new();

        // restore working dir
        env::set_current_dir(&work_item.working_dir)?;

        // get the kernel name
        let kernel_file = match work_item.kernel_file {
            Some(name) => name,
            None => {
                return Err(Box::from(
                    "[run] (profileinterface) kernel_file field missing",
                ));
            }
        };

        // for profiling we set the current target directory
        env::set_current_dir(work_item.target_dir)?;

        let kernel = fs::read_to_string(&kernel_file)?;
        // handle multiple kernel names
        let kernel_names = extract_kernel_name(kernel)?;
        log::debug!(
            "[run] (profileinterface) profile kernel_names {:#?}",
            kernel_names
        );
        if kernel_names.is_empty() {
            return Err(Box::from(
                "[run] (profileinterface) could not extract annotated kernel funcion",
            ));
        }
        let cutedsl_env_path = get_item("cutedsl_env_path")?;

        // use --set detailed rather than --set full
        for name in kernel_names.iter() {
            log::info!("[run] profiling kernel {}", name);
            let mut cmd = Command::new("sudo");
            cmd.arg("/usr/local/cuda/bin/ncu")
                .arg("--target-processes")
                .arg("all")
                .arg("--kernel-name")
                .arg(format!("regex:{}", name))
                .arg("--set")
                .arg("detailed")
                .arg("-o")
                .arg(format!("profile-{}", name))
                .arg("-f")
                .arg(format!("{}/bin/python3", cutedsl_env_path))
                .arg(kernel_file.clone());

            log::trace!("[run] (profileinterface) full cli {:?}", cmd);

            let output_res = cmd.output();

            match output_res {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    let elapsed = start.elapsed();
                    log::info!("[run] (profileinterface) completed task in {:?}", elapsed);
                    // preserve output
                    println!("{}", stdout);
                    stdout
                }
                Err(e) => {
                    log::error!("[run] (profileinterface) {}", e);
                    return Err(Box::from(e));
                }
            };

            // check if output has been compressed
            let profile_file = format!("profile-{}.ncu-repz", name);
            let profile_file_name = match Path::new(&profile_file).exists() {
                true => profile_file,
                false => format!("profile-{}.ncu-rep", name),
            };

            let convert_output_res = Command::new("sudo")
                .arg("/usr/local/cuda/bin/ncu")
                .arg("--import")
                .arg(profile_file_name)
                .arg("--page")
                .arg("details")
                .output();

            match convert_output_res {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    // check for any profiling and/or profile conversion errors
                    if stdout.contains("ERROR") {
                        return Err(Box::from(stdout));
                    }
                    let elapsed = start.elapsed();
                    log::info!(
                        "[run] (profileinterface) convert completed task in {:?}",
                        elapsed
                    );
                    // preserve output
                    println!("{}", stdout);
                    // write the final report to disk
                    fs::write(format!("{}.profile", name), stdout.clone())?;
                    profile_buffer.push('\n');
                    profile_buffer.push_str(&stdout);
                }
                Err(e) => {
                    log::error!("[run] (profileinterface) convert {}", e);
                    return Err(Box::from(e));
                }
            };
        }
        // return profile buffer with all detected kernel profiles
        Ok(profile_buffer)
    }

    fn get_elapsed_cycles(ncu_report: String) -> Result<i64, Box<dyn std::error::Error>> {
        let re = Regex::new("[\\s]{4}Elapsed\\sCycles[\\s]+cycle[\\s]+([0-9,]*)")?;
        let mut elapsed_cycles = 0;
        for cap in re.captures_iter(&ncu_report) {
            // we are only interested in the first Elapsed Cycles result in the file
            elapsed_cycles = cap[1].to_string().replace(",", "").parse::<i64>()?;
            log::trace!("[get_elapsed_cycles] {}", elapsed_cycles);
            break;
        }
        Ok(elapsed_cycles)
    }

    fn get_category(state_report: String) -> Result<String, Box<dyn std::error::Error>> {
        let re = Regex::new("[\\*]{2}PRIMARY_BOTTLENECK:[\\*]{2}[\\s]*([`a-zA-Z_]*)")?;
        let mut category = String::new();
        for cap in re.captures_iter(&state_report) {
            category = cap[1].to_string().replace("`", "").replace("_bound", "");
            log::trace!("[get_category] {}", category);
        }
        Ok(category)
    }

    fn calculate_improvement(
        baseline: i64,
        current: i64,
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let mut reward = (baseline - current) as f64 / baseline as f64;
        if reward < 0.0 {
            // add penalty for being worse
            reward += -0.1
        }
        let result = reward * 100.0;
        Ok((result, reward))
    }
}

fn extract_kernel_name(kernel: String) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut vec_res = vec![];
    let re = Regex::new("@cute.kernel[\\sdef\\s]+([a-zA-Z0-9_]*)")?;
    for cap in re.captures_iter(&kernel) {
        vec_res.push(cap[1].to_string());
    }
    Ok(vec_res)
}
