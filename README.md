# Overview

A complete cutedsl kernel reinforcement workflow written in Rust

## Usage

clone this repo

```
git clone https://github.com/lmzuccarelli/rust-cutedsl-kernel-rl

cd rust-cutedsl-kernel-rl
```

### Build binary

This assumes you have installed Rust


```
make build
```

## Prerequisites and dependencies(cuda13.2 specific)

- python
- libtorch
- cuda 13.2 and cuda 13.2 driver
- nvshmem-cuda-13
- cusparselt
- libnccl-2.30.3-1+cuda13.2 
- libnccl-devel-2.30.3-1+cuda13.2 
- libnccl-static-2.30.3-1+cuda13.2
- cudnn9-cuda-13
- nvidia nsights (ncu)
- cutlass (for CuteDSL) 

The installation of these dependencies is out of scope for this project and it's left up to the user to install.

## Workflow

There are basically 2 type of endpoint servers

- llm (for inferencing)
- gpu

The llm service can be executed on any server (no gpu required). It can use openapi type llm service, default is using claude with opus4.8

The gpu requires a gpu and relevant cuda and cutedsl dependencies to be installed and available, this is where the 'kernel' will be executed and profiled through ncu (nvidia nsights).

### Basic Flow 

- The user initiate's a workflow via the various service endpoints (via a config file) using a json payload to indicate the kernel/s to be executed and profiled.
- The gpu process is used to get an initial baseline by executing the kernel and then profiling it using **ncu** (Nvidia's profiler) for the *Elpased Cycles*.
- The llm process is prompted with results from the ncu insights (referencing the Speed of Light results), the known database of problems and recommended optimisations with the cuda-kernel code as reference.
- This process is repeated until the set trajectory value is reached (max_rollout) .
- The trajectory with the best score (lowest Elapsed Cycles as a percentage from the baseline) will be saved together with the relevant cutedsl-kernel optimized code.
- A simple workflow cli is used for the workflow controll.


## Cavaets

### Update the sudo user in the /etc/sudoers file 

Example

```bash
myuser ALL=(ALL) NOPASSWD: ALL
```

This disables asking for a password (ncu needs elevated privileges) refer to [permissions](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#AllUsersTag)

### Ensure LD_LIBRARY_PATH envar is set 

Before executing the gpu or compiler services ensure the LD_LIBRARY_PATH envar is set correctly (link to cuda lib64 and CUPTI)

Example 

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-<version>/lib64:/usr/local/cuda/extras/CUPTI/lib64
```

## Troubleshooting

### Manually test cuda kernels

Use the following cli in the specific "trajectory_XXX/step_XXX/build" directory on the gpu server (base directory is "out") on the gpu server

```bash
python3 <kernel-name>.py
```

### Manually profile cutedsl kernels

Look for the kernel name in file kernel_file.py (search for @cute.kernel annotation)

Use the following cli to execute a profile

```bash

# execute all metrics (this will include Speed of Light)
sudo ncu --target-processes all --kernel-name regex:<kernel-function_name> --set full -o profile-<kernel-name> -f /bin/python3 kernel_file.py

# convert to text format
sudo ncu --import profile-<kernel-name>.ncu-rep --page details > kernel-name.profile

```

## Replay buffer

The actual trajectory and steps within each trajectory has not been pushed into this repo, we are currently looking into creating
a S3 bucket solution
