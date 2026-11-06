use clap::{Parser, Subcommand};

/// cli struct
#[derive(Parser, Debug)]
#[command(name = env!("CARGO_PKG_NAME"))]
#[command(author = env!("CARGO_PKG_AUTHORS"))]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "CUDA KERNEL RL Agent", long_about = None)]
#[command(
    help_template = "{author-with-newline} {about-section}Version: {version} \n {usage-heading} {usage} \n {all-args} {tab}"
)]
pub struct Cli {
    #[arg(
        short,
        long,
        value_name = "config",
        help = "The config file used to configure the workflow controller, compile and gou services"
    )]
    pub config: String,
    #[command(subcommand)]
    pub command: Option<Commands>,
}
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// main control service
    Llm {},
    /// used to start gpu server (execute binary and profile kernel binary using ncu)
    Gpu {},
    Controller {},
    /// init
    Init {},
}
