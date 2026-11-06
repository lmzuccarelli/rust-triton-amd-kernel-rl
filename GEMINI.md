## Tech Stack

*   **Language**: Rust
*   **Deployment**: bash scripting


## Project Structure and Conventions

Adhere to the following structure and rules when adding or modifying code:

*   **Handlers**: All database query code must be located in the `src/handlers/` directory. Services should encapsulate business logic and data access.
*   **Cli**: Cli shcema must be located in the `src/cli/` directory. Check the struct for any problematic entries.
*   **Config**: The application must have a config file and releavnt struct to ensure minimal command line entries and must be located in the `src/config/` directory.
*   **Scripts**: The application uses this directory to "build and deploy" the service remotely. Check the bash scripot for any errors, the scipts and must be located in the `scripts` directory.
