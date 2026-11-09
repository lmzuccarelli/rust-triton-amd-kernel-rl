//! Binary entry point that regenerates `src/generated_prompts.rs`.
//!
//! It wires together the optimization-database artifacts and renders the
//! prompt source by delegating to [`write_generated_prompt`].
//!
//! Run from the crate root (so the relative `artifacts/` paths resolve):
//!
//! ```sh
//! cargo run --bin generate_prompts
//! ```

use kernel_triton_agent::prompts::generate_template::write_generated_prompt;

fn main() {
    match write_generated_prompt() {
        Ok(bytes) => {
            println!("wrote {bytes} bytes to src/inference/generated_prompts.rs");
        }
        Err(err) => {
            eprintln!("failed to generate prompts: {err}");
            std::process::exit(1);
        }
    }
}
