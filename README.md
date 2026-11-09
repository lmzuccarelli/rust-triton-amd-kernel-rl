# Overview

A complete triton kernel reinforcement workflow written in Rust for AMD gpu's

## Usage

clone this repo

```
git clone https://github.com/lmzuccarelli/rust-triton-amd-kernel-rl

cd rust-triton-amd-kernel-rl
```

### Build binary

This assumes you have installed Rust


```
make build
```

## Prerequisites and dependencies

- python
- libtorch

The installation of these dependencies is out of scope for this project and it's left up to the user to install.

## Workflow

There are basically 2 type of endpoint servers

- llm (for inferencing)
- gpu

The llm service can be executed on any server (no gpu required). It can use openapi type llm service, default is using claude with opus4.8

The gpu requires a gpu and relevant cuda and cutedsl dependencies to be installed and available, this is where the 'kernel' will be executed and profiled through ncu (nvidia nsights).

### Basic Flow 

- The user initiate's a workflow via the various service endpoints (via a config file) using a json payload to indicate the kernel/s to be executed and profiled.
- The gpu process is used to get an initial baseline by executing the kernel and then profiling it using **rocprof** (AMD's profiler) for the *Kernel Cycles*.
- The llm process is prompted with results from the AMD profile, the known database of problems and recommended optimisations with the triton-kernels code as reference.
- This process is repeated until the set trajectory value is reached (max_rollout) .
- The trajectory with the best score (lowest Kernel Cycles as a percentage from the baseline) will be saved together with the relevant triton-kernels optimized code.
- A simple workflow cli is used for the workflow controll.


## Cavaets

### Update the sudo user in the /etc/sudoers file 

Example

```bash
myuser ALL=(ALL) NOPASSWD: ALL
```

## Replay buffer

The actual trajectory and steps within each trajectory has not been pushed into this repo, we are currently looking into creating
a S3 bucket solution
