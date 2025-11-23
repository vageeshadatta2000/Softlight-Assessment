import os
import json
import time
import asyncio
import base64
from typing import List, Dict
from .browser_manager import BrowserManager
from .vision_agent import VisionAgent

class WorkflowCapturer:
    def __init__(self, output_dir: str = "output"):
        self.browser = BrowserManager(headless=False) # Visible for demo
        self.agent = VisionAgent()
        self.output_dir = output_dir
        self.history: List[str] = []
        self.captured_states: List[Dict] = []

    async def run_task(self, task: str, start_url: str):
        """
        Executes the main loop: Observe -> Think -> Act.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Starting task: {task}")
        await self.browser.start()
        
        try:
            await self.browser.navigate(start_url)
            
            step = 0
            while step < 20: # Safety limit
                step += 1
                print(f"--- Step {step} ---")
                
                # 1. Capture State
                # Wait a bit for dynamic content to load
                await asyncio.sleep(2)
                
                # Use Set-of-Mark (SoM) capture
                screenshot_b64, interactive_elements = await self.browser.capture_state_with_overlays()
                
                # Save state locally
                timestamp = int(time.time() * 1000)
                screenshot_filename = f"step_{step}_{timestamp}.jpg"
                with open(os.path.join(self.output_dir, screenshot_filename), "wb") as f:
                    f.write(base64.b64decode(screenshot_b64))
                
                # 2. Ask Agent
                decision = await self.agent.analyze_state_and_decide(task, screenshot_b64, self.history, interactive_elements)
                print(f"Agent Thought: {decision.get('thought')}")
                print(f"Agent Action: {decision.get('action')} {decision.get('params')}")
                
                # Record state metadata
                state_data = {
                    "step": step,
                    "timestamp": timestamp,
                    "screenshot": screenshot_filename,
                    "url": self.browser.page.url,
                    "agent_thought": decision.get("thought"),
                    "action_taken": decision.get("action"),
                    "action_params": decision.get("params")
                }
                self.captured_states.append(state_data)
                
                # 3. Execute Action
                action = decision.get("action")
                params = decision.get("params", {})
                
                self.history.append(f"Step {step}: {action} {params}")
                
                if action == "finish":
                    print("Task completed successfully!")
                    break
                elif action == "fail":
                    print(f"Task failed: {params.get('reason')}")
                    break
                elif action == "click":
                    selector = params.get("selector")
                    text = params.get("text")
                    idx = params.get("element_index")
                    coords = None
                    
                    # Look up coordinates if index provided
                    if idx is not None and isinstance(idx, int) and 0 <= idx < len(interactive_elements):
                        item = interactive_elements[idx]
                        # Verify it matches roughly (optional safety check)
                        coords = item.get("rect")
                        
                    await self.browser.click_element(selector=selector, text_content=text, coords=coords)
                elif action == "type":
                    selector = params.get("selector")
                    text = params.get("text")
                    await self.browser.type_text(text, selector=selector)
                elif action == "press":
                    key = params.get("key")
                    await self.browser.press_key(key)
                elif action == "scroll":
                    direction = params.get("direction", "down")
                    await self.browser.scroll(direction)
                elif action == "navigate":
                    url = params.get("url")
                    await self.browser.navigate(url)
                else:
                    print(f"Unknown action: {action}")
                    
                # Wait a bit for UI to settle
                await asyncio.sleep(2)
                
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            # Save manifest
            with open(os.path.join(self.output_dir, "manifest.json"), "w") as f:
                json.dump({"task": task, "states": self.captured_states}, f, indent=2)
            
            await self.browser.stop()
