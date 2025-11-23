import os
import json
from typing import Dict, Any, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class VisionAgent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o" 

    async def analyze_state_and_decide(self, task: str, screenshot_base64: str, previous_actions: List[str], interactive_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sends the screenshot, task, and interactive elements to the LLM to decide the next action.
        """
        
        # Format elements for prompt (limit to avoid token overflow, but 500 is safe for GPT-4o)
        elements_str = json.dumps(interactive_elements[:500], indent=2)
        
        system_prompt = f"""
        You are an automated UI agent. Your goal is to accomplish a user task on a web page.
        You will receive:
        1. The user's high-level task.
        2. A screenshot of the current page state.
        3. A list of interactive elements found in the DOM (buttons, inputs, links).
        4. A history of previous actions.

        You must output a JSON object with the following fields:
        - `thought`: A brief reasoning about the current state and what to do next.
        - `action`: The action to perform. One of: "click", "type", "press", "navigate", "finish", "fail".
        - `params`: A dictionary of parameters for the action.
            - For "click": {{"selector": "css_selector", "text": "visible text", "element_index": 0}} (Include element_index from the list if available)
            - For "type": {{"selector": "css_selector", "text": "text to type", "element_index": 0}}
            - For "press": {{"key": "Enter"}}
            - For "scroll": {{"direction": "down" or "up"}}
            - For "navigate": {{"url": "https://..."}}
            - For "finish": {{}}
            - For "fail": {{"reason": "Why it failed"}}
        
        IMPORTANT:
        - **VISUAL GROUNDING**: The screenshot has RED BOXES with NUMBERS on them. These numbers correspond to the `element_index` in the list.
        - **TO CLICK**: Look at the screenshot, find the element you want, read the NUMBER on it (e.g., "42"), and use that as `element_index`.
        - **IGNORE TEXT MATCHING**: Do not rely on text matching if the visual number is clear. The number is the ground truth.
        - **VERIFY**: If you want to click "Add Project", find the box around "Add Project" and use its number.
        
        CRITICAL HEURISTICS:
        - **AVOID GLOBAL SEARCH BARS**: Most apps have a search bar in the top/left (often with a magnifying glass icon or "Search" placeholder). DO NOT type into this unless the task is specifically to "search".
        - **CHECK PLACEHOLDERS**: When typing, look for an input with a relevant placeholder (e.g., "Project name", "Issue title") rather than a generic one.
        - **VERIFY STATE**: Before typing, ensure the modal or form you expected to open is actually visible. If not, try clicking the button again or wait.
        - **PREFER MAIN CONTENT**: For creation tasks, the relevant inputs are usually in the center of the screen or in a modal, not in the sidebar.
        
        VERIFICATION & RECOVERY:
        - **DID IT WORK?**: After every action, look at the new screenshot. Did the page change as expected?
            - *Example*: "I clicked 'Priority'. Do I see 'High', 'Medium', 'Low' now? If I see 'Current user', I clicked 'Creator' by mistake. I must retry."
        - **UNEXPECTED MODAL?**: If a modal (like "Project updates") appeared when you wanted to navigate, you clicked a status/action column instead of the name.
            - **RECOVERY**: Press "Esc" to close it, then find the **NAME** text of the item and click that specifically.
        - **WRONG CLICK?**: If a dropdown appeared, press Esc and try clicking the text title.
        - **TABLES/LISTS**: When selecting an item, ALWAYS try to click the **Text Name** of the item. Avoid clicking the center of the row if it contains status icons or other buttons.
        - **FINAL CHECK**: Before sending "finish", verify the task is ACTUALLY done. (e.g., Is the filter icon showing "1"? Is the issue actually gone?)
        - **STUCK?**: If you are stuck in a loop, try a completely different approach (e.g., use a keyboard shortcut if you know one, or search for the item).
        - USE THE PROVIDED "INTERACTIVE ELEMENTS" LIST TO GROUND YOUR ACTIONS.
        - ALWAYS include `element_index` in params if you are referring to an item from the list. This allows for robust coordinate-based clicking.
        - If the element is an icon without text, look for an `aria-label` or `id` in the list.
        - If you can't find the element in the list but see it in the screenshot, describe it visually in `thought` but try to guess a robust selector.

        """

        user_content = [
            {
                "type": "text",
                "text": f"Task: {task}\n\nPrevious Actions: {previous_actions}\n\nInteractive Elements (Top 100): {elements_str}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}",
                    "detail": "high"
                }
            }
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Error calling LLM: {{e}}")
            return {"action": "fail", "params": {"reason": str(e)}}
