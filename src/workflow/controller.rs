use crate::config::load::Parameters;
use crate::inference::generated_prompts::{
    get_optimization_plan, get_performance_state_category, get_state_prompt,
    get_task_generate_code_prompt,
};
use crate::kernel::profile::{Profile, ProfileInterface};
use crate::utils::common::{
    clean_trajectories, extract_code, extract_code_from_call, find_kernel_file,
    find_most_performant_kernel, get_trajectories, pick_weighted,
};
use crate::workflow::api_client::{process_get_call, process_post_call};
use custom_logger as log;
// Due to rate limiting executing the agent in parallel is not used for now
// use futures::stream::FuturesUnordered;
// use futures::stream::StreamExt;
use serde_derive::{Deserialize, Serialize};
use short_id::short_id_with_bytes;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;
use std::time::Instant;

pub trait ControllerInterface {
    async fn get_health(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
    async fn execute_baseline_flow(
        paramaters: Parameters,
    ) -> Result<(), Box<dyn std::error::Error>>;
    async fn execute_agent_flow(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
    fn init(paramaters: Parameters) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct Controller {}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, PartialOrd)]
pub struct OptimizationPlan {
    pub technique: String,
    pub relevance_score: f32,
    pub description: String,
}

impl ControllerInterface for Controller {
    fn init(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        for item in parameters.workflow_batch.iter() {
            let Some(name) = item.split("/").into_iter().last() else {
                return Err(Box::from("[init] workflow batch item not properly defined"));
            };
            clean_trajectories(format!(
                "{}/replay-buffer/{}/{}/rl-ncu/",
                parameters.working_dir,
                parameters.llm_model,
                name.replace(".py", "")
            ))?;
        }
        Ok(())
    }

    async fn get_health(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        // check gpu endpoint
        let res_gpu = process_get_call(format!("{}/v1/health", parameters.gpu_server_url)).await?;
        log::info!("{}", res_gpu);
        // check llm endpoint
        let res_llm = process_get_call(format!("{}/v1/health", parameters.llm_server_url)).await?;
        log::info!("{}", res_llm);
        Ok(())
    }

    async fn execute_baseline_flow(
        parameters: Parameters,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for item in parameters.workflow_batch.iter() {
            log::info!("[execute_baseline_flow] item {}", item);
            let Some(kernel_file) = item.split("/").into_iter().last() else {
                return Err(Box::from(
                    "[execute_baseline_flow] workflow batch item not properly defined",
                ));
            };
            let name = kernel_file.replace(".py", "");
            // ensure logs directory is created for each kernel
            fs::create_dir_all(format!(
                "replay-buffer/{}/{}/rl-ncu/baseline",
                parameters.llm_model, name
            ))?;

            let mut elapsed_cycles = 0_i64;
            let mut ncu_report = String::new();
            let mut state = String::new();
            let match_state = String::new();
            let mut json_plan = String::new();

            log::debug!("[execute_baseline_flow] kernel_name {}", kernel_file);
            log::debug!("[execute_baseline_flow] name {}", name);

            log::trace!(
                "[execute_baseline_flow] initial elapsed_cycles {}",
                elapsed_cycles
            );
            log::trace!(
                "[execute_baseline_flow] initial ncu_report     {}",
                ncu_report
            );
            log::trace!("[execute_baseline_flow] initial state          {}", state);
            log::trace!(
                "[execute_baseline_flow] initial match_state          {}",
                match_state
            );
            log::trace!(
                "[execute_baseline_flow] initial json_plan      {}",
                json_plan
            );

            let x = parameters.flow_control;
            let baseline_dir = format!(
                "{}/out/{}/{}/rl-ncu/baseline",
                parameters.working_dir.to_owned(),
                parameters.llm_model,
                name
            );
            log::debug!("[execute_baseline_flow] reading kernel {}", item);
            let mut code = fs::read_to_string(item)?;
            let payload = format!(
                r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_file": "{}" , "code": {:?} }}"##,
                name, parameters.working_dir, parameters.gpu_arch, baseline_dir, kernel_file, code,
            );
            // as we are saving locally to replay buffer , change name 'out' to 'replay-buffer'
            let local_baseline_dir = baseline_dir.replace("/out/", "/replay-buffer/");

            if x & 1u8 == 1 {
                // upload cutedsl kernel (item)
                log::info!("[execute_baseline_flow] baseline uploading cutedsl kernel");
                let url = format!("{}/v1/upload", parameters.gpu_server_url);
                let file_name = format!("{}/{}", local_baseline_dir, kernel_file);
                code = process_post_call(Some(file_name), url, payload.clone()).await?;
            } else {
                code = fs::read_to_string(format!("{}/{}", local_baseline_dir, kernel_file))?;
            }

            if x & 2u8 == 2 {
                let payload = format!(
                    r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_file": "{}" }}"##,
                    name, parameters.working_dir, parameters.gpu_arch, baseline_dir, kernel_file,
                );

                // call the execute endpoint
                log::info!(
                    "[execute_baseline_flow] baseline calling execute cutedsl kernel endpoint"
                );
                let url = format!("{}/v1/execute", parameters.gpu_server_url);
                let file_name = format!("{}/execute.txt", local_baseline_dir);
                process_post_call(Some(file_name), url, payload.clone()).await?;
            }

            if x & 4u8 == 4 {
                let payload = format!(
                    r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_file": "{}" }}"##,
                    name, parameters.working_dir, parameters.gpu_arch, baseline_dir, kernel_file,
                );

                // call the nvidia ncu profile endpoint
                log::info!(
                    "[execute_baseline_flow] baseline calling profile cutedsl kernel endpoint"
                );
                let url = format!("{}/v1/profile", parameters.gpu_server_url);
                let profile_file = format!("{}/profile.txt", local_baseline_dir);
                ncu_report = process_post_call(Some(profile_file), url, payload).await?;
            } else {
                let profile_file = format!("{}/profile.txt", local_baseline_dir);
                if Path::new(&profile_file).exists() {
                    ncu_report = fs::read_to_string(format!("{}/profile.txt", local_baseline_dir))?;
                }
            }

            elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
            log::info!(
                "[execute_baseline_flow] baseline elapsed_cycles {}",
                elapsed_cycles
            );

            if x & 8u8 == 8 {
                let file_name = format!("{}/llm_state_response.txt", local_baseline_dir);
                log::info!(
                    "[execute_baseline_flow] baseline calling llm endpoint (current state profile)"
                );
                // set the initial prompt for the llm
                let prompt = get_state_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                // call the llm endpoint
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                state = process_post_call(Some(file_name), url, prompt).await?;
            } else {
                let file_name = format!("{}/llm_state_response.txt", local_baseline_dir);
                if Path::new(&file_name).exists() {
                    state = fs::read_to_string(format!(
                        "{}/llm_state_response.txt",
                        local_baseline_dir
                    ))?;
                }
            }

            /*
            if x & 16u8 == 16 {
                log::info!("[execute_baseline_flow] baseline calling llm endpoint (match state)");
                // set the initial prompt for the llm
                let prompt = get_state_match_prompt(state.clone());
                // call the llm endpoint
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/llm_match_state_response.txt", local_baseline_dir);
                match_state = process_post_call(Some(file_name), url, prompt).await?;
            } else {
                let file_name = format!("{}/llm_match_state_response.txt", local_baseline_dir);
                if Path::new(&file_name).exists() {
                    match_state = fs::read_to_string(format!(
                        "{}/llm_match_state_response.txt",
                        local_baseline_dir
                    ))?;
                }
            }
            */

            if x & 32u8 == 32 {
                log::info!(
                    "[execute_baseline_flow] baseline calling llm endpoint (optimization plan)"
                );
                // get the top_n matching optimizations in json format from the llm
                // let avail_opt = get_available_optimizations();
                let prompt_op =
                    get_optimization_plan(parameters.max_rollout, state.clone(), code.clone());

                // call the llm endpoint
                let url = format!(
                    "{}/v1/prompt/{}",
                    parameters.llm_server_url,
                    parameters.llm_agent.to_string().to_lowercase()
                );
                let file_name = format!("{}/optimization-plan.json", local_baseline_dir);
                json_plan = process_post_call(Some(file_name), url, prompt_op).await?;
            } else {
                let file_name = format!("{}/optimization-plan.json", local_baseline_dir);
                if Path::new(&file_name).exists() {
                    json_plan = fs::read_to_string(format!(
                        "{}/optimization-plan.json",
                        local_baseline_dir
                    ))?;
                }
            }

            if x & 64u8 == 64 {
                let start = Instant::now();
                log::info!("[execute_baseline_flow] executing rollout");
                let json_plan_updated = json_plan.replace("```json", "");
                let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;
                let category = Profile::get_category(state)?;
                // let mut futs = FuturesUnordered::new();
                for x in 0..parameters.max_rollout {
                    // let plan = pick_weighted(plans.clone(), vec![], "".to_owned())?;
                    let plan = plans[x as usize].clone();
                    // Generate a shorter 8-character ID (6 bytes)
                    let short = short_id_with_bytes(6)?;
                    let trajectory_dir = format!(
                        "{}/replay-buffer/{}/{}/rl-ncu/trajectory_{}_{}",
                        parameters.working_dir,
                        parameters.llm_model,
                        name,
                        x + 1,
                        short
                    );
                    log::info!(
                        "[execute_baseline_flow] executing trajectory {} for technique {}",
                        trajectory_dir,
                        plan.technique.to_owned()
                    );
                    fs::create_dir_all(format!("{}/step_0", trajectory_dir.to_owned()))?;
                    let state_category = get_performance_state_category();

                    log::trace!("[execute_baseline_flow] plan {:?}", plan);
                    log::trace!("[execute_baseline_flow] state category {}", state_category);
                    log::trace!("[execute_baseline_flow] ncu report {}", ncu_report);
                    log::trace!("[execute_baseline_flow] code {}", code);

                    let task_prompt = get_task_generate_code_prompt(
                        plan.technique.to_owned(),
                        category.to_owned(),
                        plan.description.to_owned(),
                        ncu_report.to_owned(),
                        code.to_owned(),
                    );

                    fs::write(
                        format!("{}/step_0/{}.prompt", trajectory_dir, plan.technique),
                        task_prompt.clone(),
                    )?;

                    let url = format!(
                        "{}/v1/prompt/{}",
                        parameters.llm_server_url,
                        parameters.llm_agent.to_string().to_lowercase()
                    );
                    // this will always start with step 0, then within the complex flow
                    // each new step (until max.rollout) will be added
                    let prompt_file_name = format!(
                        "{}/step_0/{}_llm_response.txt",
                        trajectory_dir, plan.technique
                    );
                    let kernel_file_name =
                        format!("{}/step_0/{}.py", trajectory_dir, plan.technique);

                    // execute in parallel
                    // Due to rate limiting on the cerebras endpoints
                    // we can't execute theses calls in parallel
                    // commenting the code for now
                    //futs.push(extract_code_from_call(
                    //    Some(prompt_file_name),
                    //    kernel_file_name,
                    //    url,
                    //    task_prompt,
                    //));

                    // if this fails exit immediately
                    extract_code_from_call(
                        Some(prompt_file_name),
                        kernel_file_name,
                        url,
                        task_prompt,
                    )
                    .await?
                }

                // see the comment above
                // wait for the remaining to finish.
                // while let Some(response) = futs.next().await {
                //    match response {
                //        Ok(_) => log::info!(
                //            "[execute_baseline_flow] call (extract_code_from_call) succeeded"
                //        ),
                //        Err(e) => log::error!("[execute_baseline_flow] call failed {}", e),
                //    }
                //}

                let elapsed = start.elapsed();
                log::info!("[execute_baseline_flow] completed rollout in {:?}", elapsed);
            }
        }
        log::info!("[execute_baseline_flow] workflow controller completed successfully");
        Ok(())
    }

    async fn execute_agent_flow(parameters: Parameters) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: work out batch processing for all level1 -> level3 kernels
        for item in parameters.workflow_batch.iter() {
            // get baseline state and elapsed_cycles
            // optimization plans were calculated in the baseline run
            // no optimization plan (json) file is found in the step_0 directories
            let Some(baseline_kernel_file) = item.split("/").into_iter().last() else {
                return Err(Box::from(
                    "[execute_agent_flow] workflow batch item not properly defined",
                ));
            };

            let name = baseline_kernel_file.replace(".py", "");
            let baseline_dir = format!(
                "{}/replay-buffer/{}/{}/rl-ncu/baseline",
                parameters.working_dir, parameters.llm_model, name
            );
            log::debug!("[execute_agent_flow] baseline directory {}", baseline_dir);
            log::debug!("[execute_agent_flow] reading baseline profile");
            let baseline_ncu_report = fs::read_to_string(format!("{}/profile.txt", baseline_dir))?;
            let baseline_elapsed_cycles = Profile::get_elapsed_cycles(baseline_ncu_report.clone())?;
            let trajectories_dir = format!(
                "{}/replay-buffer/{}/{}/rl-ncu",
                parameters.working_dir, parameters.llm_model, name
            );

            let trajectories = get_trajectories(
                trajectories_dir.clone(),
                parameters.exclude_trajectories.clone(),
            )?;

            // show all trajectories
            log::debug!("[execute_agent_flow] trajectories {:#?}", trajectories);
            let mut track_fallback_kernel = format!("{}/{}", baseline_dir, baseline_kernel_file);
            let track_max_reward = 0.0;

            // from previous execution we know these kernel optimization techniques are
            // problematic (compilation, execution and/or profiling)
            let mut vec_error_techniques: Vec<String> = vec![];

            'trajectories: for current_trajectory in trajectories.iter() {
                if parameters.exclude_trajectories.contains(&"all".to_owned()) {
                    break;
                }

                log::info!("[execute_agent_flow] trajectory   : {}", current_trajectory);
                let &mut mut fallback = &mut false;
                let plan_count = parameters.max_rollout - 1;

                for step in parameters.rollout_start..parameters.max_rollout {
                    log::info!("[execute_agent_flow] current step : {}", step);
                    let local_target_dir = format!(
                        "{}/replay-buffer/{}/{}/rl-ncu/{}/step_{}",
                        parameters.working_dir,
                        parameters.llm_model,
                        name,
                        current_trajectory,
                        step
                    );
                    let target_dir = local_target_dir.replace("/replay-buffer/", "/out/");
                    log::info!(
                        "[execute_agent_flow] replay buffer directory {}",
                        local_target_dir
                    );

                    let next_target_dir = format!(
                        "{}/replay-buffer/{}/{}/rl-ncu/{}/step_{}",
                        parameters.working_dir,
                        parameters.llm_model,
                        name,
                        current_trajectory,
                        step + 1
                    );

                    if step < parameters.max_rollout - 1 {
                        let res = fs::create_dir_all(next_target_dir.clone());
                        match res {
                            Ok(_) => {
                                log::info!(
                                    "[execute_agent_flow] created next step directory {}",
                                    next_target_dir
                                );
                            }
                            Err(e) => {
                                log::error!(
                                    "[execute_agent_flow] failed to create directory {}",
                                    e
                                );
                                // this is a critical error
                                break;
                            }
                        }
                    }

                    // N.B. the error handling is purposely set to "try" not break and exit the loop
                    // If there is an error the objective is to continue to the next step
                    // Nested Ok() Err() checks have been avoided for legibility reasons

                    let mut payload = String::new();
                    let code = String::new();
                    let mut ncu_report = String::new();
                    let mut state = String::new();
                    let mut plans: Vec<OptimizationPlan> = vec![];

                    // handles flow control for fine grained tasks
                    let mut flow_control = 0u8;

                    log::debug!("[execute_agent_flow] flow controll value {flow_control}");

                    // 1. read kernel code
                    if fallback {
                        log::trace!("{payload}");
                        log::trace!("{code}");
                        log::warn!("[execute_agent_flow] fallback set");
                        log::warn!(
                            "[execute_agent_flow] track fallback kernel {track_fallback_kernel}"
                        );
                    }

                    let (current_kernel_file, current_kernel_code) = find_kernel_file(
                        local_target_dir.clone(),
                        track_fallback_kernel.to_string(),
                        &mut fallback,
                    )?;

                    log::info!(
                        "[execute_agent_flow] current kernel file {}",
                        current_kernel_file
                    );
                    log::info!("[execute_agent_flow] using path {}", local_target_dir);

                    payload = format!(
                        r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_file": "{}" , "code": {:?} }}"##,
                        name,
                        parameters.working_dir,
                        parameters.gpu_arch,
                        target_dir,
                        current_kernel_file,
                        current_kernel_code
                    );

                    // 2. upload kernel
                    log::info!(
                        "[execute_agent_flow] uploading kernel : {}",
                        current_kernel_file
                    );
                    let url = format!("{}/v1/upload", parameters.gpu_server_url);
                    let upload_res = process_post_call(None, url.clone(), payload.clone()).await;
                    match upload_res {
                        Ok(_) => {
                            log::info!("[execute_agent_flow] kernel uploaded successfully");
                            flow_control = 2u8;
                        }
                        Err(e) => {
                            log::error!("[execute_agent_flow] kernel upload error {}", e);
                            fallback = true;
                            continue;
                        }
                    }

                    // 3. execute kernel
                    if flow_control & 2u8 == 2u8 {
                        log::info!("[execute_agent_flow] calling execute cutedsl kernel endpoint");
                        let url = format!("{}/v1/execute", parameters.gpu_server_url);
                        let file_name = format!("{}/execute.txt", local_target_dir);
                        let exec_res =
                            process_post_call(Some(file_name), url, payload.clone()).await;
                        match exec_res {
                            Ok(exec) => {
                                if exec.contains("OK") {
                                    log::info!(
                                        "[execute_agent_flow] kernel execute completed successfully"
                                    );
                                    flow_control = 4u8;
                                } else {
                                    log::error!(
                                        "[execute_agent_flow] kernel execute response 'failed'"
                                    );
                                    if parameters.use_error_vec {
                                        let technique = current_kernel_file
                                            .clone()
                                            .split(".py")
                                            .next()
                                            .unwrap_or("none")
                                            .to_owned();

                                        if !vec_error_techniques.contains(&technique) {
                                            vec_error_techniques.push(technique);
                                        }
                                    }
                                    fallback = true;
                                    continue;
                                }
                            }
                            Err(e) => {
                                log::error!("[execute_agent_flow] kernel execute failed {}", e);
                                if parameters.use_error_vec {
                                    let technique = current_kernel_file
                                        .clone()
                                        .split(".py")
                                        .next()
                                        .unwrap_or("none")
                                        .to_owned();

                                    if !vec_error_techniques.contains(&technique) {
                                        vec_error_techniques.push(technique);
                                    }
                                }
                                fallback = true;
                                continue;
                            }
                        }
                    }

                    // 4. profile kernel
                    if flow_control & 4u8 == 4u8 {
                        log::info!("[execute_agent_flow] calling profile cutedsl kernel endpoint");
                        let url = format!("{}/v1/profile", parameters.gpu_server_url);
                        let file_name = format!("{}/profile.txt", local_target_dir);
                        let profile_res =
                            process_post_call(Some(file_name), url, payload.clone()).await;
                        ncu_report = match profile_res {
                            Ok(profile) => profile,
                            Err(e) => {
                                log::error!("[execute_agent_flow] kernel profile failed {}", e);
                                fallback = true;
                                continue;
                            }
                        };

                        // 5a. calculate elapsed cycles
                        let elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;

                        // 5b. calculate improvement/degradation
                        log::info!("[execute_agent_flow] elapsed cycles {}", elapsed_cycles);
                        let (perc, reward) = Profile::calculate_improvement(
                            baseline_elapsed_cycles,
                            elapsed_cycles,
                        )?;
                        log::info!(
                            "[execute_agent_flow] percentage improvement {} reward {}",
                            perc,
                            reward
                        );
                        let mut contents =
                            format!("baseline elapsed cycles : {}\n", baseline_elapsed_cycles);
                        contents
                            .push_str(&format!("elapsed cycles          : {}\n", elapsed_cycles));
                        contents.push_str(&format!("improvement             : {}%\n", perc));
                        contents.push_str(&format!("reward                  : {}\n", reward));
                        fs::write(format!("{}/stats.txt", local_target_dir), contents)?;

                        // setup fallback directory and cutedsl kernel if reward is higher than the
                        // tracking_max_reward
                        if reward > track_max_reward {
                            // TODO:
                            // track_max_reward = reward;
                            // track_fallback_kernel = format!("{}/{}", local_target_dir, kernel_file);
                            // log::warn!(
                            //    "[execute_agent_flow] setting fallback kernel to : {} with path : {}",
                            //    kernel_file,
                            //    local_target_dir
                            // );
                        }

                        if perc < parameters.degradation_threshold {
                            log::warn!("[execute_agent_flow] degradation greater than 10%");
                            if parameters.use_error_vec {
                                let technique = current_kernel_file
                                    .clone()
                                    .split(".py")
                                    .next()
                                    .unwrap_or("none")
                                    .to_owned();

                                if !vec_error_techniques.contains(&technique) {
                                    vec_error_techniques.push(technique);
                                }
                            }
                            fallback = true;
                            continue;
                        }

                        // check acceptance_threshold
                        log::debug!(
                            "[execute_agent_flow] checking threshold {} {}",
                            reward,
                            parameters.acceptance_threshold
                        );

                        if reward >= parameters.acceptance_threshold {
                            // no need to continue
                            // we have got the level of optimization to be better or equal to
                            // the acceptance_threshold criteria
                            log::info!(
                                "[execute_agent_flow] detected cutedsl kernel with reward > threshold {} exiting trajectories",
                                current_kernel_file
                            );
                            break 'trajectories;
                        }
                        flow_control = 8u8;
                    }

                    // 5. get new state from profile
                    if flow_control & 8u8 == 8u8 {
                        // get new state prompt based on the ncu_report
                        // if previous step fails it will use the baseline ncu_report
                        log::info!("[execute_agent_flow] calling llm (state)");
                        let state_prompt =
                            get_state_prompt(code.clone(), ncu_report.clone()).replace("\n", "");
                        // call the llm endpoint
                        let url = format!(
                            "{}/v1/prompt/{}",
                            parameters.llm_server_url,
                            parameters.llm_agent.to_string().to_lowercase()
                        );
                        let file_name = format!("{}/llm_state_response.txt", local_target_dir);
                        let state_res = process_post_call(Some(file_name), url, state_prompt).await;
                        state = match state_res {
                            Ok(contents) => {
                                log::info!(
                                    "[execute_agent_flow] calling llm state completed successfully"
                                );
                                flow_control = 16u8;
                                contents
                            }
                            Err(e) => {
                                log::error!("[execute_agent_flow] calling llm state failed {}", e);
                                continue;
                            }
                        };
                    }

                    // 6. get optimization plan
                    if flow_control & 16u8 == 16u8 {
                        // rate limit
                        thread::sleep(Duration::from_secs(5));

                        // get the top_n matching optimizations in json format from the llm
                        let prompt_op =
                            get_optimization_plan(plan_count, state.clone(), code.clone());
                        // call llm for optimization plan
                        log::info!("[execute_agent_flow] calling llm (optimization plan)");
                        let url = format!(
                            "{}/v1/prompt/{}",
                            parameters.llm_server_url,
                            parameters.llm_agent.to_string().to_lowercase()
                        );
                        let file_name = format!("{}/optimization-plan.json", local_target_dir);
                        let json_plan_res =
                            process_post_call(Some(file_name), url, prompt_op).await;
                        let json_plan = match json_plan_res {
                            Ok(contents) => {
                                log::info!(
                                    "[execute_agent_flow] calling llm json optimization plan completed successfully"
                                );
                                contents
                            }
                            Err(e) => {
                                log::error!(
                                    "[execute_agent_flow] calling llm json optimization plan failed {}",
                                    e
                                );
                                fallback = true;
                                continue;
                            }
                        };
                        // parse the optimization plan
                        let json_plan_updated = json_plan.replace("```json", "").replace("```", "");
                        let plans_res =
                            serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated);
                        plans = match plans_res {
                            Ok(vec_plans) => {
                                log::info!(
                                    "[execute_agent_flow] parsing optimization plan completed successfully"
                                );
                                vec_plans
                            }
                            Err(e) => {
                                log::error!(
                                    "[execute_agent_flow] parsing optimization plan failed {}",
                                    e
                                );
                                fallback = true;
                                continue;
                            }
                        };

                        // before building the complex prompt for the next step
                        // check if we have reached the max_rollout value
                        if step == parameters.max_rollout - 1 {
                            log::info!(
                                "[execute_agent_flow] max_rollout reached exiting flow gracefully"
                            );
                            break;
                        }
                        flow_control = 32u8;
                    }

                    // 7. create complex prompt and execute for the next step
                    if flow_control & 32u8 == 32u8 {
                        // if this fails it's due to regex (it will break our loop)
                        log::info!(
                            "[execute_agent_flow] setting up for the next step ({})",
                            step + 1
                        );
                        let category = Profile::get_category(state.clone())?;

                        // TODO: why?
                        let current_best_plan = track_fallback_kernel
                            .split("/")
                            .last()
                            .unwrap_or("none")
                            .to_owned();

                        let plan = pick_weighted(
                            plans.clone(),
                            vec_error_techniques.clone(),
                            current_best_plan,
                        )?;
                        log::warn!("[execute_agent_flow] selected plan {} ", plan.technique);
                        log::warn!(
                            "[execute_agent_flow] exclude plans {:?}",
                            vec_error_techniques
                        );

                        let task_prompt = get_task_generate_code_prompt(
                            plan.technique.clone(),
                            category.to_owned(),
                            plan.description,
                            ncu_report.to_owned(),
                            code.to_owned(),
                        );

                        let wr_res = fs::write(
                            format!("{}/{}.prompt", next_target_dir, plan.technique),
                            task_prompt.clone(),
                        );
                        match wr_res {
                            Ok(_) => {
                                log::info!(
                                    "[execute_agent_flow] saved {} prompt to disk ",
                                    plan.technique
                                );
                            }
                            Err(e) => {
                                log::error!("[execute_agent_flow] failed to save prompt {}", e);
                                fallback = true;
                                continue;
                            }
                        };

                        log::info!("[execute_agent_flow] calling llm (task generate code)");
                        let url = format!(
                            "{}/v1/prompt/{}",
                            parameters.llm_server_url,
                            parameters.llm_agent.to_string().to_lowercase()
                        );

                        // rate limit
                        thread::sleep(Duration::from_secs(5));

                        let file_name =
                            format!("{}/{}_llm_response.txt", next_target_dir, plan.technique);
                        let contents_res =
                            process_post_call(Some(file_name), url, task_prompt).await;
                        match contents_res {
                            Ok(contents) => {
                                // extract code
                                let code = extract_code(contents)?;
                                let wr_res = fs::write(
                                    format!("{}/{}.py", next_target_dir, plan.technique),
                                    code,
                                );
                                match wr_res {
                                    Ok(_) => {
                                        log::info!(
                                            "[execute_agent_flow] saved {} cutedsl kernel to disk",
                                            plan.technique
                                        );
                                    }
                                    Err(e) => {
                                        log::error!(
                                            "[execute_agent_flow] failed to save cutedsl kernel {}",
                                            e
                                        );
                                        fallback = true;
                                        continue;
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "[execute_agent_flow] process_post_call call generate code failed {}",
                                    e
                                );
                            }
                        }
                    }
                    track_fallback_kernel = format!("{}/{}", local_target_dir, current_kernel_file);
                }
                // end of rollout (step) loop
            }
            // finally find the optimal cutedsl kernel
            if parameters.create_stats {
                find_most_performant_kernel(trajectories_dir)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // this brings everything from parent's scope into this scope
    use crate::config::load::{ConfigInterface, ImplConfigInterface};
    use regex::Regex;
    use std::fs;

    #[tokio::test]
    async fn test_all() -> Result<(), Box<dyn std::error::Error>> {
        let impl_config = ImplConfigInterface {};
        let res_params = impl_config.read("config/application-config.json".to_owned());
        let parameters = match res_params {
            Ok(params) => params,
            Err(e) => {
                eprintln!("[main] config file error: {}", e);
                std::process::exit(1);
            }
        };

        println!("[main] testing parameters {:#?}", parameters);

        /*
        let item = "level1/001_Square_matrix_multiplication";
        let model = "gpt-oss-120b";
        println!("[execute_baseline_flow] testing current flow control");
        let ncu_report = fs::read_to_string(format!(
            "logs/{}/{}/rl-ncu/baseline/profile.txt",
            model, item
        ))?;
        let elapsed_cycles = Profile::get_elapsed_cycles(ncu_report.clone())?;
        println!(
            "[execute_baseline_flow] testing elapsed_cycles {}",
            elapsed_cycles
        );
        let (result, reward) = Profile::calculate_improvement(elapsed_cycles, 7677773)?;
        println!(
            "[execute_baseline_flow] testing result {} : reward {}",
            result, reward
        );

        let (result, reward) = Profile::calculate_improvement(38724415, 38732739)?;
        println!(
            "[execute_baseline_flow] testing result {} : reward {}",
            result, reward
        );

        let json_plan = fs::read_to_string(format!(
            "logs/{}/{}/rl-ncu/trajectory_1_mINMOfqW/step_0/optimization-plan.json",
            model, item
        ))?;
        let state = fs::read_to_string(format!(
            "logs/{}/{}/rl-ncu/baseline/llm_state_response.txt",
            model, item
        ))?;

        let category = Profile::get_category(state)?;
        println!("[execute_baseline_flow] testing category {}", category);

        println!();

        let json_plan_updated = json_plan.replace("```json", "").replace("```", "");
        let plans = serde_json::from_str::<Vec<OptimizationPlan>>(&json_plan_updated)?;

        println!();

        for _x in 0..3 {
            let pick = pick_weighted(plans.clone(), vec![], "".to_owned())?;
            println!(
                "[execute_baseline_flow] testing weighted plan {} {}",
                pick.technique, pick.relevance_score
            );
        }

        let base_dir = format!(
            "{}/logs/{}/{}/rl-ncu/trajectory_1_Po9t6Fh_/step_0",
            parameters.working_dir, model, item
        );

        let (file, kernel) = find_kernel_file(base_dir.clone(), "".to_string(), &mut false)?;
        println!("[run] cutedsl kernel file {}", file);
        println!("[run] cutedsl kernel len {}", kernel.len());
        //let updated = cuda_kernel.chars().filter(|c| c.is_ascii()).collect();

        let payload = format!(
            r##"{{ "name": "{}", "working_dir": "{}", "gpu_arch": "{}" , "target_dir": "{}", "kernel_name": "{}" , "code": {:?} }}"##,
            item, parameters.working_dir, parameters.gpu_arch, base_dir, file, kernel
        );

        let work_item_res = serde_json::from_slice::<WorkItem>(payload.as_bytes())?;
        println!("[run] work items {:?}", work_item_res);
        */

        //let cuda_kernel = fs::read_to_string("tests/tensor_core_utilization.cu")?;
        let kernel = fs::read_to_string("tests/square_gemm.py")?;
        let re = Regex::new("@cute.kernel[\\sdef\\s]+([a-zA-Z0-9_]*)")?;
        let mut kernel_name;
        for cap in re.captures_iter(&kernel) {
            kernel_name = cap[1].to_string();
            println!("[run] profiling kernel {}", kernel_name);
            assert_eq!("sgemm_kernel", kernel_name);
        }

        /*

        let vec_trajectories = get_trajectories(
            format!("{}/logs/{}/{}/rl-ncu", parameters.working_dir, model, item),
            vec![],
        )?;
        println!("{:?}", vec_trajectories);

        find_most_performant_kernel(format!(
            "{}/logs/{}/{}/rl-ncu",
            parameters.working_dir, model, item
        ))?;
        */

        Ok(())
    }
}
