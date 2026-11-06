use crate::config::load::WorkItem;
use custom_logger as log;
use std::env;
use std::fs;
use std::path::Path;

pub trait FileInterface {
    async fn kernel_rw(
        work_item: WorkItem,
        write: bool,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

pub struct FileOperation {}

impl FileInterface for FileOperation {
    async fn kernel_rw(
        work_item: WorkItem,
        write: bool,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // restore working dir
        env::set_current_dir(work_item.working_dir)?;

        let kernel_code;
        let dir = work_item.target_dir.clone();
        log::debug!("[kernel_rw] directory {}", dir);

        match work_item.kernel_file {
            Some(kernel_file) => {
                let file = format!("{}/{}", dir, kernel_file);
                if !Path::new(&dir).exists() {
                    log::debug!("[kernel_rw] creating dir {}", dir);
                    fs::create_dir_all(&dir)?;
                }
                log::debug!("[kernel_rw] kernel file {}", file);
                if write && work_item.code.is_some() {
                    kernel_code = work_item.code.unwrap_or("".to_string());
                    log::trace!("[kernel_rw] code {}", kernel_code);
                    fs::write(file, kernel_code.clone())?;
                } else {
                    kernel_code = fs::read_to_string(&file)?;
                }
            }
            None => {
                return Err(Box::from(
                    "[kernel_rw] (fileinterface) kernel_file field missing",
                ));
            }
        }
        Ok(kernel_code)
    }
}
