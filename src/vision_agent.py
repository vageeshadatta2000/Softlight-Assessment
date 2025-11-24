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
        2. A screenshot of the current page state with RED BOXES and NUMBERS labeling interactive elements.
        3. A list of interactive elements found in the DOM (buttons, inputs, links).
        4. A history of previous actions.

        You must output a JSON object with the following fields:
        - `thought`: A brief reasoning about the current state and what to do next.
        - `action`: The action to perform. One of: "click", "type", "press", "scroll", "navigate", "finish", "fail".
        - `params`: A dictionary of parameters for the action.
            - For "click": {{"selector": "css_selector", "text": "exact visible text", "element_index": 0}}
            - For "type": {{"selector": "css_selector", "text": "text to type", "element_index": 0}}
            - For "press": {{"key": "Enter"}}
            - For "scroll": {{"direction": "down" or "up"}}
            - For "navigate": {{"url": "https://..."}}
            - For "finish": {{}}
            - For "fail": {{"reason": "Why it failed"}}

        ## CRITICAL: ELEMENT SELECTION PROCESS

        **Step 1 - LOCATE**: Find the element you want in the screenshot. Look at the RED BOX around it.

        **Step 2 - READ NUMBER**: Read the NUMBER on that specific red box carefully. Numbers can be small and overlap - zoom in mentally if needed.

        **Step 3 - VERIFY IN LIST**: Look up that index in the interactive elements list. Check if the `text` field matches what you expect.
            - Example: If you want to click "Priority" and see box #42, check that elements[42].text contains "Priority"
            - If it doesn't match, you read the wrong number! Look again at the screenshot.

        **Step 4 - INCLUDE TEXT**: ALWAYS include the `text` parameter with the EXACT text you see on the element.
            - This serves as a safety check - if the index is wrong, the system can find the right element by text.

        ## VISUAL GROUNDING RULES

        - **NUMBERS ARE INDICES**: The red number on each box = the `element_index` to use
        - **VERIFY BEFORE CLICKING**: State in your thought: "I see element [X] labeled '[text]' at index [N]"
        - **DENSE UIs**: In menus/dropdowns, boxes may overlap. Look carefully at which number is ON the element you want.
        - **WHEN UNCERTAIN**: If you can't clearly read the number, use the element list to find by text, then use that index.

        ## AVOIDING COMMON MISTAKES

        - **ADJACENT ELEMENTS**: "Creator" and "Priority" may be next to each other in a menu. Their indices are different! Don't confuse them.
        - **NESTED ELEMENTS**: Parent and child elements both get boxes. Click the most specific one (usually has the text).
        - **ICONS**: If clicking an icon, look for elements with `aria-label` in the list.

        ## HEURISTICS

        - **AVOID GLOBAL SEARCH BARS**: Don't type in the top search bar unless the task is specifically to search.
        - **CHECK PLACEHOLDERS**: For typing, find inputs with relevant placeholders (e.g., "Project name", "Title", "Name").
        - **PREFER MAIN CONTENT**: Forms/inputs are usually in the center or in modals, not sidebars.
        - **SEMANTIC FIELD MATCHING**: Match fields to task semantics - "name/title" → title fields (usually top, larger), "content/body" → main text areas (below title), "description" → description fields.
        - **VISUAL HIERARCHY**: Titles are typically larger and at top. Body/content areas are below. Look at font size and position to distinguish.
        - **AUTO-FOCUSED FIELDS - CRITICAL**: When creating new items (pages, projects, tasks), the name/title field is ALREADY FOCUSED with a blinking cursor.
            - If you see placeholder text like "Project name", "Untitled", "Task name" - the cursor is ALREADY THERE
            - DO NOT click any element. Just use "type" action with NO element_index
            - Set element_index to null/omit it to type into the currently focused field
            - Example: {{"action": "type", "params": {{"text": "My Project Name"}}}} - no selector, no element_index
        - **LOOK FOR PLACEHOLDERS**: Input fields show placeholder text like "Project name", "Untitled", "Task name". If you see this placeholder, the field is likely already focused.

        ## VERIFICATION & RECOVERY

        - **DID IT WORK?**: After each action, CAREFULLY examine the new screenshot to verify the expected change occurred.
            - Did a modal/dialog/menu appear after clicking?
            - Did text appear in the input field after typing?
            - Did the UI state change in the expected way?
        - **IF UI UNCHANGED**: Your action FAILED. Do NOT proceed - retry with a different approach.
        - **WRONG CLICK?**: Press Esc to close any unexpected dropdown/modal, then retry with correct element.
        - **TABLES/LISTS**: Click the text NAME of items, not status icons or empty row areas.
        - **STUCK IN LOOP?**: Try a different approach - keyboard shortcut, or search for the item.

        ## CRITICAL: STATE AWARENESS

        In your `thought`, ALWAYS describe:
        1. **What you see NOW** in the current screenshot (what's visible, what's open)
        2. **What changed** from the previous state (or if nothing changed)
        3. **What you expect to happen** from your next action

        ## CRITICAL: FINISH CONDITIONS

        **NEVER use "finish" based on assumptions. Only finish when you can VISUALLY CONFIRM:**
        - The requested item/object EXISTS in the current screenshot
        - The requested state/change is VISIBLE in the UI
        - You can point to specific visual evidence the task is complete

        **COMMON MISTAKES - DO NOT:**
        - Say "I've previously done X" if you cannot see evidence of X in the CURRENT screenshot
        - Assume an action worked without seeing the result
        - Finish after opening a menu/sidebar - you must also SELECT the option and APPLY it
        - Trust your memory of previous actions - only trust what you SEE NOW

        **For filter/sort tasks:** You must see the filter APPLIED (e.g., "Priority: Medium" chip visible, or filtered results showing)
        **For create tasks:** You must see the created item in the list/view

        **If you cannot see proof of completion, DO NOT finish.** Instead:
        - Investigate what went wrong
        - Retry failed steps
        - Or use "fail" with explanation

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
