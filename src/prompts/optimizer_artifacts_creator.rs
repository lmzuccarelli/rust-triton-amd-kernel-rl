//! Builds human-readable summaries from the optimization-database JSON file.
//!
//! [`create_known_states`] renders every optimization strategy with its
//! primary bottleneck, secondary characteristics, and available optimizations.
//!
//! [`create_available_optimizations`] renders a more compact per-state listing
//! of just the available optimizations, including each technique's predicted
//! speedup, confidence score, and description.

use std::fmt::Write as _;
use std::fs;

use serde_json::Value;

/// Default location of the optimization database.
pub const DEFAULT_PATH: &str = "artifacts/optimization_database.json";

/// Reads the optimization database at `path` and returns the rendered
/// "known optimization states" summary as a string.
///
/// Used in function get_stage_match_prompt
pub fn create_known_states(path: &str) -> Result<String, String> {
    let contents =
        fs::read_to_string(path).map_err(|err| format!("could not read {path}: {err}"))?;

    let database: Value =
        serde_json::from_str(&contents).map_err(|err| format!("invalid JSON in {path}: {err}"))?;

    let strategies = database
        .get("optimization_strategies")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing 'optimization_strategies' object in {path}"))?;

    let mut buffer = String::new();

    for (state_name, state) in strategies {
        let _ = writeln!(buffer, "STATE: {state_name}");
        let _ = writeln!(
            buffer,
            "Primary Bottleneck: {}",
            state
                .get("primary_bottleneck")
                .and_then(Value::as_str)
                .unwrap_or("")
        );
        let _ = writeln!(
            buffer,
            "Secondary Characteristics: {}",
            state
                .get("secondary_characteristics")
                .and_then(Value::as_str)
                .unwrap_or("")
        );
        let _ = writeln!(buffer, "Available Optimizations:");

        if let Some(optimizations) = state.get("optimizations").and_then(Value::as_array) {
            for optimization in optimizations {
                let technique = optimization
                    .get("technique")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>");
                let speedup = optimization
                    .get("predicted_speedup")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                let _ = writeln!(buffer, "  - {technique}: {speedup:.2}x speedup");
            }
        }

        buffer.push('\n');
    }

    Ok(buffer)
}

/// Reads the optimization database at `path` and returns a Markdown summary
/// grouping each performance state with its characteristics, primary
/// bottleneck, and available optimizations.
///
/// Each state is rendered as a `#### State: <name>` header followed by
/// `**Characteristics**`, `**Primary Bottleneck**`, and an `**Optimizations**`
/// list, where every optimization is a line in the form
/// `- **<technique>**: <speedup>x predicted speedup - <description>`.
pub fn create_performance_state_category(path: &str) -> Result<String, String> {
    let contents =
        fs::read_to_string(path).map_err(|err| format!("could not read {path}: {err}"))?;

    let database: Value =
        serde_json::from_str(&contents).map_err(|err| format!("invalid JSON in {path}: {err}"))?;

    let strategies = database
        .get("optimization_strategies")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing 'optimization_strategies' object in {path}"))?;

    let mut buffer = String::new();

    for (state_name, state) in strategies {
        let _ = writeln!(buffer, "#### State: {state_name}");
        let _ = writeln!(
            buffer,
            "**Characteristics**: {}",
            state
                .get("secondary_characteristics")
                .and_then(Value::as_str)
                .unwrap_or("")
        );
        let _ = writeln!(
            buffer,
            "**Primary Bottleneck**: {}",
            state
                .get("primary_bottleneck")
                .and_then(Value::as_str)
                .unwrap_or("")
        );
        let _ = writeln!(buffer, "**Optimizations**:");

        if let Some(optimizations) = state.get("optimizations").and_then(Value::as_array) {
            for optimization in optimizations {
                let technique = optimization
                    .get("technique")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>");
                let speedup = optimization
                    .get("predicted_speedup")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                let description = optimization
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let _ = writeln!(
                    buffer,
                    "- **{technique}**: {speedup:.2}x predicted speedup - {description}"
                );
            }
        }

        buffer.push('\n');
    }

    Ok(buffer)
}

/// Reads the optimization database at `path` and returns a compact listing of
/// the available optimizations for each state.
///
/// Each state is rendered as a `STATE: <name>` header followed by one line per
/// optimization in the form
/// `  - <technique> (pred <speedup>x | conf <confidence>): <description>`.
pub fn create_available_optimizations(path: &str) -> Result<String, String> {
    let contents =
        fs::read_to_string(path).map_err(|err| format!("could not read {path}: {err}"))?;

    let database: Value =
        serde_json::from_str(&contents).map_err(|err| format!("invalid JSON in {path}: {err}"))?;

    let strategies = database
        .get("optimization_strategies")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing 'optimization_strategies' object in {path}"))?;

    let mut buffer = String::new();

    for (state_name, state) in strategies {
        let _ = writeln!(buffer, "STATE: {state_name}");

        if let Some(optimizations) = state.get("optimizations").and_then(Value::as_array) {
            for optimization in optimizations {
                let technique = optimization
                    .get("technique")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>");
                let speedup = optimization
                    .get("predicted_speedup")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0);
                let confidence = optimization
                    .get("confidence_score")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let description = optimization
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let _ = writeln!(
                    buffer,
                    "  - {technique} (pred {speedup:.2}x | conf {confidence}): {description}"
                );
            }
        }

        buffer.push('\n');
    }

    Ok(buffer)
}
