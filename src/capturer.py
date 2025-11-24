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
            consecutive_same_action = 0
            last_action_key = None

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

                # Loop detection - check if repeating same action
                current_action_key = f"{action}_{params.get('text', '')}_{params.get('element_index', '')}"
                if current_action_key == last_action_key:
                    consecutive_same_action += 1
                    if consecutive_same_action >= 3:
                        print(f"WARNING: Detected loop - same action repeated {consecutive_same_action} times. Breaking.")
                        print("Agent is stuck. Consider a different approach.")
                        break
                else:
                    consecutive_same_action = 1
                    last_action_key = current_action_key

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
                        item_text = item.get("text", "").lower()

                        # Verify the element text matches what the agent intended
                        if text and text.lower() not in item_text and item_text not in text.lower():
                            print(f"WARNING: Index {idx} has text '{item.get('text')}', expected '{text}'")
                            print("Attempting to find correct element by text match...")

                            # Search for the correct element by text - prefer smaller/more specific elements
                            matches = []
                            for i, el in enumerate(interactive_elements):
                                el_text = el.get("text", "").lower()
                                if text.lower() in el_text or el_text in text.lower():
                                    el_area = el['rect']['width'] * el['rect']['height']
                                    matches.append((i, el, el_area))

                            if matches:
                                # Sort by area (smallest first) to get most specific element
                                matches.sort(key=lambda x: x[2])
                                idx, item, _ = matches[0]
                                print(f"Found {len(matches)} matches, using smallest: index {idx}: '{item.get('text')}'")
                            else:
                                print(f"No text match found, using original index {idx}")

                        coords = item.get("rect")
                        el_area = coords['width'] * coords['height']

                        # If clicking a very small element (likely an icon), try to find a better match by text
                        if el_area < 500 and text and len(text) > 3:
                            print(f"WARNING: Element {idx} is very small ({el_area:.0f}px²) for text '{text}'")
                            print("Searching for larger element with matching text...")

                            better_matches = []
                            # Search for elements containing the full text first
                            full_text = text.lower()

                            for i, el in enumerate(interactive_elements):
                                el_text = el.get("text", "").lower()
                                el_aria = (el.get("ariaLabel") or "").lower()
                                combined_text = el_text + " " + el_aria

                                # Only match if the element text contains most of our search text
                                # or vice versa (avoid matching unrelated items)
                                if full_text in combined_text or combined_text in full_text:
                                    area = el['rect']['width'] * el['rect']['height']
                                    if area > 500:  # Only consider reasonably sized elements
                                        better_matches.append((i, el, area))

                            if better_matches:
                                # Use smallest of the reasonably-sized matches
                                better_matches.sort(key=lambda x: x[2])
                                idx, item, _ = better_matches[0]
                                coords = item.get("rect")
                                el_area = coords['width'] * coords['height']
                                print(f"Found better match: index {idx}: '{item.get('text')}'")
                            else:
                                print(f"No better match found, using original small element")

                        print(f"Clicking element {idx}: '{item.get('text', 'N/A')[:30]}' at ({coords['x']:.0f}, {coords['y']:.0f}) size: {coords['width']:.0f}x{coords['height']:.0f} ({el_area:.0f}px²)")

                    cursor_screenshot = await self.browser.click_element(selector=selector, text_content=text, coords=coords)

                    # Save screenshot with cursor showing where we clicked
                    if cursor_screenshot:
                        cursor_filename = f"step_{step}_cursor_{int(time.time() * 1000)}.png"
                        with open(os.path.join(self.output_dir, cursor_filename), "wb") as f:
                            f.write(cursor_screenshot)
                        print(f"Saved cursor screenshot: {cursor_filename}")

                    # Brief pause then capture post-click state for verification
                    await asyncio.sleep(0.3)
                    post_click_screenshot = await self.browser.get_screenshot_base64()
                    post_click_filename = f"step_{step}_click_result_{int(time.time() * 1000)}.jpg"
                    with open(os.path.join(self.output_dir, post_click_filename), "wb") as f:
                        f.write(base64.b64decode(post_click_screenshot))
                    print(f"Saved post-click screenshot: {post_click_filename}")
                elif action == "type":
                    selector = params.get("selector")
                    text = params.get("text")
                    idx = params.get("element_index")

                    # If NO element_index provided, type directly into currently focused field
                    # This is for auto-focused fields like "Project name" in modals
                    if idx is None:
                        print(f"Typing '{text}' into currently focused field (no element specified)...")
                        await self.browser.page.keyboard.type(text)
                    # If element_index provided, click to focus first, then type directly
                    elif isinstance(idx, int) and 0 <= idx < len(interactive_elements):
                        item = interactive_elements[idx]
                        coords = item.get("rect")
                        if coords:
                            print(f"Clicking to focus element {idx}: '{item.get('text', 'N/A')[:30]}' before typing")
                            await self.browser.click_element(coords=coords)
                            await asyncio.sleep(0.3)
                            # Type directly with keyboard since we clicked to focus
                            print(f"Typing '{text}' into focused element...")
                            await self.browser.page.keyboard.type(text)
                    else:
                        # Invalid index, use selector-based typing
                        await self.browser.type_text(text, selector=selector)

                    # Capture post-type screenshot to verify text was entered
                    await asyncio.sleep(0.3)
                    post_type_screenshot = await self.browser.get_screenshot_base64()
                    post_type_filename = f"step_{step}_type_result_{int(time.time() * 1000)}.jpg"
                    with open(os.path.join(self.output_dir, post_type_filename), "wb") as f:
                        f.write(base64.b64decode(post_type_screenshot))
                    print(f"Saved post-type screenshot: {post_type_filename}")
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
