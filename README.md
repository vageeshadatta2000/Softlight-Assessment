# UI State Capture Agent

An AI-powered agent that navigates web applications to perform tasks and captures the UI state (screenshots and metadata) at each step.

## Setup

1.  **Install Dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    playwright install chromium
    ```

2.  **Configure API Key**:
    Copy `.env.example` to `.env` and add your OpenAI API Key:
    ```bash
    cp .env.example .env
    # Edit .env and set OPENAI_API_KEY=sk-...
    ```

## Usage

Run the agent with a task description and a starting URL:

```bash
python main.py --task "Search for 'Artificial Intelligence' and click the first result" --url "https://en.wikipedia.org" --output "output/wiki_test"
```

## Output

The agent will create a directory (default: `output/run_X`) containing:
-   `step_N_TIMESTAMP.jpg`: Screenshots of each state.
-   `manifest.json`: A JSON file describing the workflow, including agent reasoning, actions taken, and metadata for each step.

## Architecture

-   `src/browser_manager.py`: Handles Playwright browser automation (navigation, clicking, typing, screenshots).
-   `src/vision_agent.py`: Interfaces with GPT-4o to analyze screenshots and decide the next action.
-   `src/capturer.py`: The main loop that orchestrates the observation-reasoning-action cycle.
