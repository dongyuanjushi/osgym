import logging
import io
import base64
from PIL import Image
from datetime import datetime
import concurrent.futures
import re

from .utils.utils import encode_screenshot

from .utils.call_llm import call_llm_with_single_response

from typing import Dict, Any, List, Tuple, Union

import json

from .utils.qwen_vl_utils import smart_resize

from datetime import datetime

ACTION_N = 1

description_prompt = (
    "Use a mouse and keyboard to interact with the computer GUI and take "
    "screenshots to complete the user's task."
)

action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
"""

tools_def = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": description_prompt,
        "parameters": {
            "properties": {
                "action": {
                    "description": action_description_prompt,
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "triple_click",
                        "scroll",
                        "hscroll",
                        "wait",
                        "terminate",
                        "answer",
                    ],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.", "type": "string"},
                "coordinate": {"description": "(x, y) coordinates.", "type": "array"},
                "pixels": {"description": "Scroll amount.", "type": "number"},
                "time": {"description": "Seconds to wait.", "type": "number"},
                "status": {
                    "description": "Task status for terminate.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
            "type": "object",
        },
    },
}


SYSTEM_PROMPT_QWEN35_VL = """"Use a mouse and keyboard to interact with a computer, and take screenshots. 
This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.

You may call the tools defined below to assist with the given task.

Here are some tips for using the tools:
- Use a mouse and keyboard to interact with a computer, and take screenshots.",
- This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
- Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
- The screen's resolution is 1000x1000."
- Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
- If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
- Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",

You have access to the following functions:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "parameters": <args-json-object>}
</tool_call>

## Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}. 
""" + f"""## Action Rules:
- The system is running on a x86_64 ubuntu system.
- Chrome is the default browser that have been installed for you to use.
- The current working directory is /home/user.
- The password for the user is "password". Use it when you need to authenticate or use sudo commands.
- The current date is {datetime.now().strftime("%Y-%m-%d")}.
- Execute exactly one action per interaction.
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- Leave all windows and applications open after completing the task.
- If finishing, use action=terminate in the tool call. 
    - Issue the status as success if the task is completed successfully
    - Issue the status as failure if the task is infeasible to complete due to environment constraints. 
"""

import io
from PIL import Image
from io import BytesIO

def _to_screenshot_bytes(screenshot: Union[bytes, str]) -> bytes:
    """Normalize a screenshot to raw bytes.

    The OSGym FastAPI server returns screenshots as base64-encoded strings
    in `/reset` and `/step` responses, while the in-process DesktopEnv
    yields raw bytes. Accept either so the agent works against both.
    """
    if isinstance(screenshot, bytes):
        return screenshot
    if isinstance(screenshot, str):
        if screenshot.startswith("data:"):
            screenshot = screenshot.split(",", 1)[1]
        return base64.b64decode(screenshot)
    raise TypeError(f"Unsupported screenshot type: {type(screenshot)}")


def process_image(image_bytes):
    """
    Process an image for Qwen VL models (thinking variant).
    Uses a tighter resize cap consistent with the thinking DUN agent.
    """
    image_bytes = _to_screenshot_bytes(image_bytes)
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800
        # max_pixels=1280*720
    )

    image = image.resize((resized_width, resized_height))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return processed_bytes

class Qwen35VLAgent:
    def __init__(
        self, 
        screen_size: tuple, approach: str, 
        policy_model: str, policy_model_provider: str, policy_model_endpoint: str, 
        logger: logging.Logger
    ):
        self.screen_size = screen_size
        self.approach = approach
        self.policy_model = policy_model
        self.policy_model_provider = policy_model_provider
        self.policy_model_endpoint = policy_model_endpoint

        self.history_window_size = 2
        self.logger = logger
        self.coordinate_type = "relative"

    def construct_messages(self, instruction, obs: Dict[str, Any]):
        base64_screenshot = encode_screenshot(process_image(obs["screenshot"]))
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_QWEN35_VL
            },
            {
                "role": "user",
                "content": f"""Please generate the next move according to the UI screenshot, instruction and previous actions.
Instruction: {instruction}. """
            }
        ]

        assert len(self.history) == len(self.screenshots), "The number of history and screenshots should be the same"

        start_idx = max(0, len(self.history) - self.history_window_size)

        self.logger.info(f"Last {min(len(self.history), self.history_window_size)} action histories: {self.history[start_idx:]}")

        for i in range(start_idx, len(self.history)):
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.screenshots[i]
                        }
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"{self.history[i]}"
                    }
                ]
            })
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_screenshot
                    }
                }
            ]
        })
        self.screenshots.append(base64_screenshot)

        return messages

    def adjust_coordinates(
        self,
        x: float, y: float,
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ):
        if not (original_width and original_height):
            return int(x), int(y)
        if self.coordinate_type == "absolute":
            # scale from processed pixels to original
            if processed_width and processed_height:
                x_scale = original_width / processed_width
                y_scale = original_height / processed_height
                return int(x * x_scale), int(y * y_scale)
            return int(x), int(y)
        # relative: scale from 0..999 grid
        x_scale = original_width / 999
        y_scale = original_height / 999
        return int(x * x_scale), int(y * y_scale)

    def process_tool_call(
        self, 
        tool_call: Dict[str, Any],
        original_width: int = None,
        original_height: int = None,
        processed_width: int = None,
        processed_height: int = None,
    ):
        pyautogui_code: List[str] = []

        try:
            args = tool_call["parameters"]
            
            action = args["action"]

            if action == "left_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    pyautogui_code.append(f"pyautogui.click({adj_x}, {adj_y})")
                else:
                    pyautogui_code.append("pyautogui.click()")

            elif action == "right_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    pyautogui_code.append(
                        f"pyautogui.rightClick({adj_x}, {adj_y})"
                    )
                else:
                    pyautogui_code.append("pyautogui.rightClick()")

            elif action == "middle_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    pyautogui_code.append(
                        f"pyautogui.middleClick({adj_x}, {adj_y})"
                    )
                else:
                    pyautogui_code.append("pyautogui.middleClick()")

            elif action == "double_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    pyautogui_code.append(
                        f"pyautogui.doubleClick({adj_x}, {adj_y})"
                    )
                else:
                    pyautogui_code.append("pyautogui.doubleClick()")
            
            elif action == "triple_click":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    pyautogui_code.append(
                        f"pyautogui.tripleClick({adj_x}, {adj_y})"
                    )
                else:
                    pyautogui_code.append("pyautogui.tripleClick()")

            elif action == "type":
                text = args.get("text", "")
                pyautogui_code.append(f"pyautogui.typewrite('{text}')")

            elif action == "key":
                keys = args.get("keys", [])
                if isinstance(keys, list):
                    cleaned_keys = []
                    for key in keys:
                        if isinstance(key, str):
                            if key.startswith("keys=["):
                                key = key[6:]
                            if key.endswith("]"):
                                key = key[:-1]
                            if key.startswith("['") or key.startswith('["'):
                                key = key[2:] if len(key) > 2 else key
                            if key.endswith("']") or key.endswith('"]'):
                                key = key[:-2] if len(key) > 2 else key
                            key = key.strip()
                            cleaned_keys.append(key)
                        else:
                            cleaned_keys.append(key)
                    keys = cleaned_keys

                keys_str = ", ".join([f"'{key}'" for key in keys])
                if len(keys) > 1:
                    pyautogui_code.append(f"pyautogui.hotkey({keys_str})")
                else:
                    pyautogui_code.append(f"pyautogui.press({keys_str})")

            elif action == "scroll":
                pixels = args.get("pixels", 0)
                pyautogui_code.append(f"pyautogui.scroll({pixels})")

            elif action == "hscroll":
                pixels = args.get("pixels", 0)
                pyautogui_code.append(f"pyautogui.hscroll({pixels})")

            elif action == "wait":
                pyautogui_code.append("WAIT")

            elif action == "terminate":
                # pyautogui_code.append("DONE")
                if "status" in args:
                    if args["status"] == "success":
                        pyautogui_code.append("DONE")
                    else:
                        pyautogui_code.append("FAIL")

            elif action == "mouse_move":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    pyautogui_code.append(
                        f"pyautogui.moveTo({adj_x}, {adj_y})"
                    )
                else:
                    pyautogui_code.append("pyautogui.moveTo(0, 0)")

            elif action == "left_click_drag":
                if "coordinate" in args:
                    x, y = args["coordinate"]
                    adj_x, adj_y = self.adjust_coordinates(x, y, original_width, original_height, processed_width, processed_height)
                    duration = args.get("duration", 0.5)
                    pyautogui_code.append(
                        f"pyautogui.dragTo({adj_x}, {adj_y}, duration={duration})"
                    )
                else:
                    pyautogui_code.append("pyautogui.dragTo(0, 0)")
                    
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse tool call: {e}")
        
        if len(pyautogui_code) > 0:
            if "DONE" in pyautogui_code or "WAIT" in pyautogui_code or "FAIL" in pyautogui_code:
                return pyautogui_code[0]
            else:
                pyautogui_code.insert(0, "import pyautogui")
                return "; ".join(pyautogui_code)

        return pyautogui_code

    def parse_action_and_tool_call(self, response_str: str):
        thought = ""
        action = None

        # Use regex to extract the thought between 'Action: ' and '<tool_call>'
        thought_match = re.search(r'Action:\s*(.*?)(?=<tool_call>)', response_str, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Use regex to extract the content between <tool_call> and </tool_call>
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response_str, re.DOTALL)
        if tool_call_match:
            tool_call_content = tool_call_match.group(1).strip()
            try:
                action = json.loads(tool_call_content)
            except Exception as e:
                self.logger.error(f"Failed to parse action JSON: {e}")
                action = None

        return thought, action

    def parse_response(
        self,
        response_list: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Parse LLM response and convert it to low level action and pyautogui code.
        """
        action_candidates: List[Dict[str, Any]] = []

        for response in response_list:
            low_level_instruction, tool_call = self.parse_action_and_tool_call(response)
            action_candidates.append({
                "thought": low_level_instruction,
                "action": tool_call
            })

        return action_candidates

    def generate_single_response(self, messages: List[Dict[str, Any]], temperature: float = 0.8):
        response = call_llm_with_single_response(
            llm_config={
                "model": self.policy_model,
                "provider": self.policy_model_provider,
                "endpoint": self.policy_model_endpoint
            },
            messages=messages,
            max_tokens=4000,
            temperature=temperature
        )
        return response

    def generate_responses_in_parallel(self, messages: List[Dict[str, Any]], temperature: float = 0.8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=ACTION_N) as executor:
            import random
            random.seed(42)
            futures = [executor.submit(self.generate_single_response, messages, random.uniform(temperature, 1.0)) for _ in range(ACTION_N)]
            results = [future.result() for future in futures]
            return results

    def predict(self, instruction, obs):
        messages = self.construct_messages(instruction, obs)
        retry_times = 3
        temperature = 0.7
        while True:
            if retry_times <= 0:
                return "", "", "FAIL"
            
            try:
                responses = self.generate_responses_in_parallel(messages, temperature)
                self.logger.info(f"Raw responses: {responses}")

                observation = ""
                action_candidates = self.parse_response(responses)

                self.logger.info(f"Action candidates: {action_candidates}")
                best_idx = 0
                
                thought = action_candidates[best_idx]["thought"]
                action = action_candidates[best_idx]["action"]
                action_code = self.process_tool_call(
                    tool_call=action,
                    original_width=self.screen_size[0],
                    original_height=self.screen_size[1]
                )

                history_response = responses[best_idx]

                self.history.append(f"{history_response}")
                break

            except Exception as e:
                self.logger.error(f"Error in calling LLM: {e}")
                retry_times -= 1
                # temperature = min(temperature + 0.1, 1.0)
                continue

        return observation, thought, action_code

    def parse_sections(self, response_str: str):
        observation = ""
        observation_match = re.search(r"<observation>(.*?)</observation>", response_str, re.DOTALL | re.IGNORECASE)
        if observation_match:
            observation = observation_match.group(1).strip()
        else:
            observation = ""

        thought_pattern = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)
        action_pattern = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)

        thought_matches = thought_pattern.findall(response_str)
        action_matches = action_pattern.findall(response_str)

        action_candidates = []
        for t, a in zip(thought_matches, action_matches):
            action_candidates.append({
                "thought": t.strip(),
                "action": a.strip()
            })

        return observation, action_candidates

    def generate_rollouts_in_parallel(self, action_candidates, imagination_horizon):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(action_candidates)) as executor:
            futures = [executor.submit(self.generate_single_rollout, action_candidate, imagination_horizon) for action_candidate in action_candidates]
            results = [future.result() for future in futures]
            return results

    def reset(self, result_dir: str = None):
        self.result_dir = result_dir
        self.history = []
        self.screenshots = []