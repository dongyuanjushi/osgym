import openai

from typing import Optional, Any, List, Dict, Any, Union

import json

import re

from pydantic import BaseModel

import time

import litellm

import base64

import cv2

import numpy as np

import ast

def parse_json_response(response: str, response_format: BaseModel = None) -> Union[Dict[str, Any], None]:
    """
    Parse JSON response from LLM that may contain various formats.
    
    Args:
        response: Raw response string from LLM
        response_format: Expected Pydantic model format
    
    Returns:
        Parsed JSON dictionary or None if parsing fails
    """
    if not response or not response.strip():
        return None
    
    # Clean the response
    response = response.strip()
    
    # Method 1: Try direct JSON parsing
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Method 2: Extract JSON from code blocks
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # ```json { ... } ```
        r'```\s*(\{.*?\})\s*```',      # ``` { ... } ```
        r'`(\{.*?\})`',                # `{ ... }`
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                return parsed
            except json.JSONDecodeError:
                continue
    
    # Method 3: Find JSON objects in the text
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            # Validate it has expected keys if response_format is provided
            if response_format:
                required_fields = response_format.model_fields.keys()
                if all(field in parsed for field in required_fields):
                    return parsed
            else:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Method 4: Try to extract key-value pairs manually
    # if response_format:
    #     try:
    #         result = {}
    #         for field_name in response_format.model_fields.keys():
    #             patterns = [
    #                 rf'["\']?{field_name}["\']?\s*:\s*["\']([^"\']*)["\']',
    #                 rf'{field_name}[:\s]+([^\n,}}]+)',
    #             ]
                
    #             for pattern in patterns:
    #                 match = re.search(pattern, response, re.IGNORECASE)
    #                 if match:
    #                     result[field_name] = match.group(1).strip()
    #                     break
            
    #         if len(result) == len(response_format.model_fields):
    #             return result
    #     except Exception:
    #         pass
    
    return None

def encode_numpy_image_to_base64(image: np.ndarray) -> str:
    """Converts a numpy array image to base64 string.
    
    Args:
        image: Numpy array representing an image (height, width, channels)
        
    Returns:
        Base64 encoded string of the image
    """
    # Convert numpy array to bytes
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Failed to encode image to png format")
    
    # Convert bytes to base64 string
    image_bytes = buffer.tobytes()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return "data:image/jpeg;base64," + base64_string

def encode_image_bytes(image_content):
    base64_str = base64.b64encode(image_content).decode('utf-8')
    return "data:image/jpeg;base64," + base64_str

def encode_screenshot(screenshot: Union[bytes, str]) -> str:
    if isinstance(screenshot, bytes):
        base_64_str = base64.b64encode(screenshot).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    elif isinstance(screenshot, str):
        bytes_obj = bytes_literal_to_bytesio(screenshot)
        base_64_str = base64.b64encode(bytes_obj).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    else:
        raise ValueError("type of screenshot is not supported, only bytes or str is supported")

def bytes_literal_to_bytesio(bytes_literal_str):
    bytes_obj = ast.literal_eval(bytes_literal_str)

    if not isinstance(bytes_obj, bytes):
        raise ValueError("not a valid bytes literal")

    return bytes_obj