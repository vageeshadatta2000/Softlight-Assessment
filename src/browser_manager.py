import base64
import asyncio
import os
import io
from typing import List, Dict, Any, Optional
from playwright.async_api import async_playwright, Page, Browser, BrowserContext, ElementHandle
from PIL import Image, ImageDraw, ImageFont

class BrowserManager:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def start(self):
        """Starts the Playwright browser with a persistent context."""
        self.playwright = await async_playwright().start()
        
        # Use a persistent context to save login session
        user_data_dir = os.path.abspath("user_data")
        os.makedirs(user_data_dir, exist_ok=True)
        
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir,
            headless=self.headless,
            channel="chrome",
            viewport={"width": 1280, "height": 720},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
            ],
            ignore_default_args=["--enable-automation"],
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        self.browser = None # Persistent context doesn't expose a browser object in the same way

    async def stop(self):
        """Stops the browser."""
        if self.context:
            await self.context.close()
        # Browser is implied in persistent context
        if self.playwright:
            await self.playwright.stop()

    async def navigate(self, url: str):
        """Navigates to a URL."""
        if not self.page:
            raise RuntimeError("Browser not started")
        await self.page.goto(url)
        # Notion/Linear are heavy SPAs, networkidle might never happen due to background polling
        # Relaxing to domcontentloaded to avoid timeouts
        await self.page.wait_for_load_state("domcontentloaded")
        await self.wait_for_ui_stability()

    async def wait_for_ui_stability(self):
        """Waits for the UI to settle by checking for common layout elements."""
        if not self.page:
            return
        
        try:
            # Wait for at least one interactive element to appear
            # or specific common containers (nav, main, sidebar)
            await self.page.wait_for_selector("button, a, input, [role='button'], nav, main", timeout=10000)
            # Small extra buffer for animations
            await asyncio.sleep(1)
        except:
            print("Warning: UI stability check timed out, proceeding anyway...")

    async def get_screenshot_base64(self) -> str:
        """Captures a screenshot and returns it as a base64 string."""
        if not self.page:
            raise RuntimeError("Browser not started")
        # Switch to PNG to resolve API error
        screenshot_bytes = await self.page.screenshot(type="png")
        return base64.b64encode(screenshot_bytes).decode("utf-8")

    async def get_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Scans the page (including frames and shadow DOM) for interactive elements.
        """
        if not self.page:
            raise RuntimeError("Browser not started")
        
        # Complex script to traverse frames and shadow roots
        elements = await self.page.evaluate("""async () => {
            const items = [];
            // Broad selector for modern apps
            const selector = 'button, a, input, textarea, select, [role="button"], [role="link"], [role="checkbox"], [role="menuitem"], [role="menuitemcheckbox"], [role="option"], [role="tab"], [role="treeitem"], [tabindex], li, div[onclick], span[onclick], svg';

            function isVisible(el) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 5 && rect.height > 5 && style.visibility !== 'hidden' && style.display !== 'none' && style.opacity !== '0';
            }

            function processElement(el) {
                if (isVisible(el)) {
                    let text = el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '';
                    if (!text && el.tagName.toLowerCase() === 'svg') {
                         text = el.getAttribute('aria-label') || 'icon';
                    }
                    text = (text || "").replace(/\\s+/g, ' ').trim();
                    
                    // Heuristic: if it has text or is an input/button/link, keep it
                    if (text || ['input', 'textarea', 'select', 'button', 'a'].includes(el.tagName.toLowerCase()) || el.getAttribute('role')) {
                        const rect = el.getBoundingClientRect();
                        items.push({
                            index: items.length,
                            tagName: el.tagName.toLowerCase(),
                            text: text.substring(0, 50),
                            id: el.id,
                            placeholder: el.getAttribute('placeholder'),
                            type: el.getAttribute('type'),
                            ariaLabel: el.getAttribute('aria-label'),
                            role: el.getAttribute('role'),
                            rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}
                        });
                    }
                }
            }

            // 1. Standard Query (Fast)
            document.querySelectorAll(selector).forEach(el => processElement(el));

            // 2. Shadow DOM Traversal (Deep)
            function traverse(root) {
                // Only traverse if we haven't found much, or just do it anyway? 
                // Let's just look for shadow roots specifically.
                const allEls = root.querySelectorAll('*');
                allEls.forEach(el => {
                    if (el.shadowRoot) {
                        el.shadowRoot.querySelectorAll(selector).forEach(shadowEl => processElement(shadowEl));
                        traverse(el.shadowRoot);
                    }
                });
            }
            traverse(document);
            
            return items;
        }""")
        
        # Also scan frames (Playwright handles this better from Python side usually, but let's try simple first)
        # Note: Cross-origin frames might block access, but for same-origin (Notion) it should work.
        
        print(f"Found {len(elements)} interactive elements.")
        return elements

    async def capture_state_with_overlays(self) -> (str, List[Dict[str, Any]]):
        """
        Captures screenshot and elements, then draws Set-of-Mark overlays (boxes + IDs).
        Returns (annotated_screenshot_base64, interactive_elements).
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        # 1. Get elements and raw screenshot
        all_elements = await self.get_interactive_elements()
        screenshot_bytes = await self.page.screenshot(type="png")

        # 2. Filter to only visible, drawable elements to ensure index sync
        # This is CRITICAL: only elements that will be drawn get indices
        visible_elements = []
        for el in all_elements:
            rect = el['rect']
            x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
            # Must be on-screen and have reasonable size
            if x >= 0 and y >= 0 and w > 5 and h > 5:
                visible_elements.append(el)

        # 3. Remove highly overlapping boxes (OmniParser approach: 90% IoU threshold)
        # This prevents parent/child elements from both being drawn
        def compute_iou(rect1, rect2):
            """Compute Intersection over Union between two rectangles."""
            x1, y1, w1, h1 = rect1['x'], rect1['y'], rect1['width'], rect1['height']
            x2, y2, w2, h2 = rect2['x'], rect2['y'], rect2['width'], rect2['height']

            # Intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0

            inter_area = (xi2 - xi1) * (yi2 - yi1)

            # Union
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area

            return inter_area / union_area if union_area > 0 else 0.0

        def compute_containment(rect1, rect2):
            """Check how much rect1 is contained within rect2."""
            x1, y1, w1, h1 = rect1['x'], rect1['y'], rect1['width'], rect1['height']
            x2, y2, w2, h2 = rect2['x'], rect2['y'], rect2['width'], rect2['height']

            # Intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0

            inter_area = (xi2 - xi1) * (yi2 - yi1)
            area1 = w1 * h1

            return inter_area / area1 if area1 > 0 else 0.0

        # Sort by area (smaller elements first - prefer more specific elements)
        visible_elements.sort(key=lambda el: el['rect']['width'] * el['rect']['height'])

        # Filter out elements with high overlap
        filtered_elements = []
        for el in visible_elements:
            should_keep = True
            el_area = el['rect']['width'] * el['rect']['height']

            # Always keep input/textarea elements - they're critical for typing
            is_input = el.get('tagName') in ['input', 'textarea'] or el.get('role') in ['textbox', 'searchbox']

            for kept_el in filtered_elements:
                iou = compute_iou(el['rect'], kept_el['rect'])
                containment = compute_containment(el['rect'], kept_el['rect'])

                # Remove if >90% IoU (OmniParser threshold) or almost fully contained
                # BUT always keep inputs even if contained
                if (iou > 0.9 or containment > 0.95) and not is_input:
                    should_keep = False
                    break

            if should_keep:
                # Skip very large elements (likely containers) unless they're inputs
                # Threshold: 30% of viewport (1280x720 = 921600, 30% = ~276000)
                if el_area > 250000 and not is_input:
                    continue
                filtered_elements.append(el)

        visible_elements = filtered_elements

        # Re-index elements to match visual labels
        for i, el in enumerate(visible_elements):
            el['index'] = i

        # 3. Open image with PIL
        image = Image.open(io.BytesIO(screenshot_bytes))
        draw = ImageDraw.Draw(image)

        # Load a font (try default, fallback to simple)
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        # 4. Draw overlays - now indices match exactly
        for i, el in enumerate(visible_elements):
            rect = el['rect']
            x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']

            # Draw bounding box (Red)
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

            # Draw ID tag (Red background, White text)
            tag_text = str(i)

            # Calculate text size using getbbox
            text_bbox = font.getbbox(tag_text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            tag_w = text_width + 6
            tag_h = text_height + 6

            # Position tag at top-left of box, but keep inside image
            tag_x = x
            tag_y = y - tag_h if y > tag_h else y

            draw.rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + tag_h], fill="red")
            draw.text((tag_x + 3, tag_y + 3), tag_text, fill="white", font=font)

        # 5. Save to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print(f"Labeled {len(visible_elements)} elements (filtered from {len(all_elements)} detected, removed overlaps via IoU).")

        return img_str, visible_elements

    async def highlight_element(self, selector: str):
        """Highlights an element for visual debugging."""
        if not self.page:
            return
        try:
            await self.page.eval_on_selector(selector, "el => el.style.border = '2px solid red'")
        except:
            pass

    async def _show_visual_cursor(self, x: float, y: float):
        """Injects a visual cursor (red dot) at the coordinates."""
        if not self.page:
            return
        await self.page.evaluate(f"""
            () => {{
                const cursor = document.createElement('div');
                cursor.style.position = 'absolute';
                cursor.style.left = '{x}px';
                cursor.style.top = '{y}px';
                cursor.style.width = '20px';
                cursor.style.height = '20px';
                cursor.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
                cursor.style.borderRadius = '50%';
                cursor.style.pointerEvents = 'none';
                cursor.style.zIndex = '10000';
                cursor.style.transform = 'translate(-50%, -50%)';
                cursor.style.boxShadow = '0 0 10px rgba(255, 0, 0, 0.5)';
                cursor.style.transition = 'all 0.2s ease';
                document.body.appendChild(cursor);
                
                // Remove after a short delay
                setTimeout(() => cursor.remove(), 1000);
            }}
        """)

    async def click_element(self, text_content: str = None, selector: str = None, coords: Dict[str, float] = None):
        """Clicks an element. Prioritizes coordinates (SoM) if available.
        Returns screenshot with cursor if coords provided, else None."""
        if not self.page:
            raise RuntimeError("Browser not started")

        # 1. Priority: Coordinate Click (Visual Grounding / Set-of-Mark)
        # If the agent provided an index/coords, it chose a specific red box on the screenshot.
        # We trust this exact position over a potentially generic selector like "button".
        if coords:
            print(f"Executing SoM Click at {coords}...")
            try:
                x = coords["x"] + coords["width"] / 2
                y = coords["y"] + coords["height"] / 2

                await self._show_visual_cursor(x, y)
                await asyncio.sleep(0.2) # Brief pause to show cursor

                # Capture screenshot with cursor visible before clicking
                cursor_screenshot = await self.page.screenshot(type="png")

                await self.page.mouse.click(x, y)
                await asyncio.sleep(0.5) # Wait for UI response
                await self.page.wait_for_load_state("domcontentloaded")
                return cursor_screenshot
            except Exception as e:
                print(f"Coordinate click failed: {e}. Falling back to selector...")

        # 2. Fallback: Selector/Text Click
        locator = None
        if selector:
            locator = self.page.locator(selector).first
        elif text_content:
            locator = self.page.get_by_text(text_content).first
        
        if locator:
            try:
                # Wait for element to be attached and visible
                await locator.wait_for(state="visible", timeout=3000)
                
                # Highlight
                await locator.evaluate("el => el.style.outline = '3px solid red'")
                
                # Get position for cursor
                box = await locator.bounding_box()
                if box:
                    center_x = box["x"] + box["width"] / 2
                    center_y = box["y"] + box["height"] / 2
                    await self._show_visual_cursor(center_x, center_y)
                
                await asyncio.sleep(0.5)
                await locator.click()
                await self.page.wait_for_load_state("domcontentloaded")
                return
            except Exception as e:
                print(f"Locator click failed: {e}")
        
        raise ValueError("Could not click: No valid coordinates or selector worked.")

    async def type_text(self, text: str, selector: str = None):
        """Types text into an input with visual feedback."""
        if not self.page:
            raise RuntimeError("Browser not started")

        locator = None
        if selector:
            locator = self.page.locator(selector).first

        if locator:
            try:
                # Wait for element
                await locator.wait_for(state="visible", timeout=5000)

                # Highlight (Blue for typing)
                await locator.evaluate("el => el.style.outline = '3px solid blue'")

                # Visual cursor
                box = await locator.bounding_box()
                if box:
                    center_x = box["x"] + box["width"] / 2
                    center_y = box["y"] + box["height"] / 2
                    await self._show_visual_cursor(center_x, center_y)

                await asyncio.sleep(0.5)

                # Try fill first, then fallback to keyboard type for contenteditable
                try:
                    await locator.fill(text)
                except Exception as fill_error:
                    print(f"Fill failed (likely contenteditable): {fill_error}")
                    print("Using keyboard.type() instead...")
                    # Clear existing content and type
                    await locator.click()
                    await self.page.keyboard.press("Control+a")
                    await self.page.keyboard.type(text)
            except Exception as e:
                print(f"Type failed: {e}")
                print("Attempting blind typing (assuming focus)...")
                await self.page.keyboard.type(text)
        else:
            # Blind type - element should already be focused from click
            print("No selector provided, typing with keyboard...")
            await self.page.keyboard.type(text)

    async def press_key(self, key: str):
        if not self.page:
            raise RuntimeError("Browser not started")
        await self.page.keyboard.press(key)
        await self.page.wait_for_load_state("networkidle")

    async def scroll(self, direction: str = "down", amount: int = 500):
        """Scrolls the page."""
        if not self.page:
            raise RuntimeError("Browser not started")
        
        if direction == "down":
            await self.page.mouse.wheel(0, amount)
        elif direction == "up":
            await self.page.mouse.wheel(0, -amount)
        
        await asyncio.sleep(1) # Wait for scroll animation
