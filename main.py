import asyncio
import argparse
import os
from src.capturer import WorkflowCapturer

async def main():
    parser = argparse.ArgumentParser(description="AI UI State Capture Agent")
    parser.add_argument("--task", type=str, required=True, help="The task description")
    parser.add_argument("--url", type=str, required=True, help="The starting URL")
    parser.add_argument("--output", type=str, default="output/run_1", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in .env or export it.")
        return

    capturer = WorkflowCapturer(output_dir=args.output)
    await capturer.run_task(args.task, args.url)

if __name__ == "__main__":
    asyncio.run(main())
