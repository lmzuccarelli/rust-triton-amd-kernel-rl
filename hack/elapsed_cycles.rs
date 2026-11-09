use regex::Regex;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::Path;

// Theoretical hardware ceilings for AMD Instinct MI300X (gfx942)
const MI300X_PEAK_BANDWIDTH_BYTES_PER_NS: f64 = 5300.0;
const MI300X_TOTAL_COMPUTE_UNITS: f64 = 304.0;
const MI300X_MAX_WAVES_PER_CU: f64 = 32.0;
const MI300X_MAX_CONCURRENT_WAVES: f64 = MI300X_TOTAL_COMPUTE_UNITS * MI300X_MAX_WAVES_PER_CU; // 9,728

#[derive(Debug, Deserialize)]
struct ProfileRow {
    #[serde(rename = "KernelName")]
    kernel_name: String,
    #[serde(rename = "DurationNs")]
    duration_ns: f64,
    #[serde(rename = "SQ_CYCLES")]
    sq_cycles: u64,
    #[serde(rename = "SQ_WAVES")]
    sq_waves: f64,
    #[serde(rename = "VRAM_RD_BYTES")]
    vram_rd_bytes: f64,
    #[serde(rename = "VRAM_WR_BYTES")]
    vram_wr_bytes: f64,
}

#[derive(Debug)]
pub struct DeepKernelMetrics {
    pub kernel_name: String,
    pub duration_ns: f64,
    pub core_sq_cycles: u64,
    pub total_waves_launched: f64,
    pub memory_speed_of_light_pct: f64,
    pub wavefront_occupancy_pct: f64,
}

pub fn analyze_triton_profile<P: AsRef<Path>>(
    csv_path: P,
    target_kernel_substring: &str,
) -> Result<Vec<DeepKernelMetrics>, Box<dyn Error>> {
    let file = File::open(csv_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut metrics_list = Vec::new();
    let re = Regex::new(target_kernel_substring)?;

    for result in rdr.deserialize() {
        let record: ProfileRow = result?;

        if re.is_match(&record.kernel_name) {
            let total_bytes = record.vram_rd_bytes + record.vram_wr_bytes;

            // 1. Memory SOL Calculation
            let achieved_bandwidth = total_bytes / record.duration_ns;
            let memory_sol = (achieved_bandwidth / MI300X_PEAK_BANDWIDTH_BYTES_PER_NS) * 100.0;

            // 2. Wavefront Occupancy Calculation
            // Compares actual launched wavefronts against total execution slot capacity
            let occupancy = (record.sq_waves / MI300X_MAX_CONCURRENT_WAVES) * 100.0;

            metrics_list.push(DeepKernelMetrics {
                kernel_name: record.kernel_name,
                duration_ns: record.duration_ns,
                core_sq_cycles: record.sq_cycles,
                total_waves_launched: record.sq_waves,
                memory_speed_of_light_pct: memory_sol,
                wavefront_occupancy_pct: occupancy.min(100.0), // Cap at 100% mathematically
            });
        }
    }

    Ok(metrics_list)
}

fn main() {
    let csv_file_path = "./profile_output/profile.csv";

    match analyze_triton_profile(csv_file_path, "add_kernel") {
        Ok(metrics) => {
            for m in metrics {
                println!("==================================================");
                println!("Kernel Name:            {}", m.kernel_name);
                println!("Duration:               {:.2} ns", m.duration_ns);
                println!("Core Execution Cycles:  {} cycles", m.core_sq_cycles);
                println!(
                    "Total Wavefronts:       {:.0} waves",
                    m.total_waves_launched
                );
                println!("--------------------------------------------------");
                println!(
                    "Memory Speed of Light:  {:.2}%",
                    m.memory_speed_of_light_pct
                );
                println!("Wavefront Occupancy:    {:.2}%", m.wavefront_occupancy_pct);
            }
        }
        Err(e) => eprintln!("Failed parsing profile: {}", e),
    }
}

/*
NOTES

1. If Wavefront Occupancy is low (< 30%) and Memory SOL is low: The Triton kernel grid does not have enough independent blocks to saturate the huge 304-CU MI300X fabric.
   Action: The agent should increase the input data array or re-evaluate the execution grid size configuration.

2. If Wavefront Occupancy is high (> 70%) but Memory SOL is low: Your original unoptimized baseline BLOCK = 64 configuration will hit exactly this state.
   The GPU is technically filled with waves, but those waves are constantly stalling because they request tiny, fragmented memory slices.
   Action: The agent must increase BLOCK (e.g., 256, 512, 1024) to widen and coalesce the global memory data packets into single, streamlined transactions.

ROCm Docs
Accelerator and GPU hardware specifications
Accelerator and GPU hardware specifications. Accelerator and GPU hardware specifications

*/
