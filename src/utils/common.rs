use crate::MAP_LOOKUP;
use crate::workflow::api_client::process_post_call;
use crate::workflow::controller::OptimizationPlan;
use custom_logger as log;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use regex::Regex;
use std::fs;
use std::thread;
use std::time::Duration;
use walkdir::WalkDir;

// common helper functions

pub fn extract_code(input: String) -> Result<String, Box<dyn std::error::Error>> {
    let start = input.find("```python").unwrap_or(0);
    let start = start + 9;
    let end = input[start..].find("```").unwrap_or(0);
    let end = start + end;
    let result = &input[start..end];
    Ok(result.to_string())
}

pub fn clean_trajectories(base_dir: String) -> Result<(), Box<dyn std::error::Error>> {
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_file() {
                    let file = obj.path().to_string_lossy().to_string();
                    // don't delete .prompt and .cu files
                    if (file.contains("trajectory_") && file.contains("step_0"))
                        && (file.contains(".txt") || file.contains("json"))
                    {
                        log::trace!("[clean_trajectories] removing file {}", file);
                        fs::remove_file(&file)?;
                    }
                }
                // remove all step directories greater than 0
                if obj.path().is_dir() {
                    let dir = obj.path().to_string_lossy().to_string();
                    if dir.contains("trajectory_")
                        && dir.contains("step_")
                        && !dir.contains("step_0")
                    {
                        log::debug!("[clean_trajectories] removing directories {}", dir);
                        fs::remove_dir_all(&dir)?;
                    }
                }
            }
            Err(e) => {
                log::error!("[clean_trajectories] error : {}", e);
            }
        }
    }
    Ok(())
}

pub fn find_most_performant_kernel(base_dir: String) -> Result<(), Box<dyn std::error::Error>> {
    let re = Regex::new("reward[\\s]*:\\s([0-9\\.-]+)")?;
    let mut max_reward = 0.0;
    let mut kernel_path = String::new();
    let mut reward_values = String::new();
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_file() {
                    let file = obj.path().to_string_lossy();
                    if file.contains("stats.txt") {
                        let contents_res = fs::read_to_string(file.to_string());
                        match contents_res {
                            Ok(contents) => {
                                for cap in re.captures_iter(&contents) {
                                    let reward = cap[1].to_string().parse::<f64>()?;
                                    if reward > -0.6 {
                                        reward_values.push_str(&format!("{},", reward));
                                    }
                                    if reward > max_reward {
                                        max_reward = reward;
                                        kernel_path = file.to_string();
                                    }
                                    log::debug!("[find_most_performant_kernel] reward {}", reward);
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "[find_most_performant_kernel] error reading stats file{}",
                                    e
                                );
                            }
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("[find_most_performant_kernel] error : {}", e);
            }
        }
    }
    let vec_parts: Vec<&str> = kernel_path.split("/").collect();
    let updated_path = vec_parts[..vec_parts.len() - 1].join("/").to_string();
    let (kernel_name, kernel_contents) =
        find_kernel_file(updated_path.clone(), "".to_string(), &mut false)?;
    let result_msg = format!(
        "[find_most_performant_kernel] found kernel {} in path {} with reward {}",
        kernel_name, updated_path, max_reward
    );
    log::info!("{result_msg}");
    let write_dir: Vec<&str> = updated_path.split("rl-ncu").collect();
    fs::write(
        format!("{}/rl-ncu/final_rl_cutedsl_perf.py", write_dir[0]),
        kernel_contents,
    )?;
    fs::write(
        format!("{}/rl-ncu/rewards.csv", write_dir[0]),
        // exclude trailing ","
        &reward_values[0..reward_values.len() - 1],
    )?;
    fs::write(format!("{}/rl-ncu/results.txt", write_dir[0]), result_msg)?;
    Ok(())
}

pub fn pick_weighted(
    mut plans: Vec<OptimizationPlan>,
    exclude: Vec<String>,
    current_best_plan: String,
) -> Result<OptimizationPlan, Box<dyn std::error::Error>> {
    let mut pick = OptimizationPlan {
        technique: current_best_plan,
        relevance_score: 0.9,
        description: "".to_string(),
    };
    // first exclude plans that we know don't work
    plans.retain(|x| !exclude.contains(&x.technique));

    // not used
    let weights = plans
        .clone()
        .iter()
        .map(|x| x.relevance_score.powi(3))
        .collect::<Vec<f32>>();

    // sort by relevance_score
    plans.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if weights.len() > 0 {
        log::trace!("[pick_weighted] {:?}", weights);

        // not used
        let dist = WeightedIndex::new(weights)?;
        let mut rng = rand::rng();
        let index = dist.sample(&mut rng);
        log::trace!("[pick_weighted] using index {}", index);
        // end not used

        // use the first item (asc sorted)
        pick = plans[0].clone();
    }
    log::debug!("[pick_weighted] selecting technique : {}", pick.technique);
    Ok(pick)
}

pub fn find_kernel_file(
    dir: String,
    fallback_kernel: String,
    fallback: &mut bool,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let mut kernel = String::new();
    let mut file = String::new();
    log::debug!("[find_kernel_file] fallback {}", *fallback);
    if !*fallback {
        let files = fs::read_dir(dir.clone())?;
        for f in files {
            match f {
                Ok(name) => {
                    let cf = name.file_name().to_string_lossy().to_string();
                    if cf.contains(".py") {
                        // find the first kernel file in the directory
                        file = cf.clone();
                        let contents = fs::read_to_string(format!("{}/{}", dir, cf))?;
                        kernel = contents.chars().filter(|c| c.is_ascii()).collect();
                        break;
                    }
                }
                Err(e) => {
                    return Err(Box::from(format!("[find_kernel_file] error {}", e)));
                }
            }
        }
    }
    if *fallback {
        log::debug!("[find_cuda_file] current directory : {}", dir);
        log::debug!("[find_cuda_file] fallback using : {}", fallback_kernel);
        let contents = fs::read_to_string(&fallback_kernel)?;
        file = fallback_kernel.split("/").last().unwrap_or("").to_string();
        kernel = contents.chars().filter(|c| c.is_ascii()).collect();
        log::debug!("[find_cuda_file] copying fallback kernel");
        fs::copy(fallback_kernel, format!("{}/{}", dir, file))?;
        *fallback = false;
    }
    Ok((file, kernel))
}

pub async fn extract_code_from_call(
    prompt_file_name: Option<String>,
    kernel_file_name: String,
    url: String,
    data: String,
) -> Result<(), Box<dyn std::error::Error>> {
    log::trace!(
        "[extract_code_from_call] prompt file name {:?}",
        prompt_file_name
    );
    log::trace!(
        "[extract_code_from_call] kernel file name {}",
        kernel_file_name
    );
    let contents = process_post_call(prompt_file_name, url, data).await?;
    let code = extract_code(contents)?;
    fs::write(kernel_file_name, code)?;
    log::debug!("[extract_code_from_call] delaying thread (limit requests)");
    thread::sleep(Duration::from_secs(5));
    Ok(())
}

pub fn get_trajectories(
    base_dir: String,
    exclude_trajectories: Vec<String>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut vec_trajectories = vec![];
    let re = Regex::new("(trajectory_[0-9]+_[0-9a-zA-Z_-]*)$")?;
    for e in WalkDir::new(base_dir) {
        match e {
            Ok(obj) => {
                if obj.path().is_dir() {
                    let file = obj.path().to_string_lossy();
                    for cap in re.captures_iter(&file) {
                        let trajectory = cap[1].to_string();
                        if !exclude_trajectories.contains(&trajectory) {
                            vec_trajectories.push(cap[1].to_string());
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("[get_trajectories] error : {}", e);
            }
        }
    }
    Ok(vec_trajectories)
}

pub fn get_item(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let hm_guard = MAP_LOOKUP.lock().map_err(|_| "mutex lock failed")?;
    let value = match hm_guard.as_ref() {
        Some(res) => {
            let item_value = res.get(name);
            match item_value {
                Some(final_value) => final_value,
                None => {
                    return Err(Box::from(format!(
                        "[get_item] hashmap lookup {} not found",
                        name
                    )));
                }
            }
        }
        None => {
            return Err(Box::from("[get_item] error validating hashmap lookup"));
        }
    };
    Ok(value.to_string())
}
