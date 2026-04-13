from typing import Optional, Any, List, Dict, Any, Union

import openai

import time

def call_llm_with_single_response(
    messages: List[Dict[str, str]], 
    llm_config: Dict[str, Any], 
    max_tokens: int = 1000, 
    temperature: float = 0.0, 
    top_p: float = 1.0, 
    response_format: Optional[Any] = None
) -> str:
    model = llm_config["model"]
    provider = llm_config["provider"]
    if provider == "bedrock":
        assert "endpoint" in llm_config, "endpoint is required for bedrock"
        litellm_client = openai.OpenAI(base_url=llm_config["endpoint"], api_key="token-123")
        while True:
            try:
                if response_format is not None:
                    completion = litellm_client.beta.chat.completions.parse(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        response_format=response_format # TODO: add response_format back
                    )
                else:
                    completion = litellm_client.chat.completions.create(
                        model=f"{provider}/{model}",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        extra_headers={
                            "anthropic-beta": "computer-use-2025-01-24"
                        }
                    )
                break

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)
                continue
            
        return completion.choices[0].message.content
        
    elif provider == "vllm":
        assert "endpoint" in llm_config, "endpoint is required for vllm"
        vllm_client = openai.OpenAI(
            base_url=llm_config["endpoint"],
            api_key="token-abc123",
        )
        if response_format is not None:
            completion = vllm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body={"guided_json": response_format.model_json_schema()}
            )
        else:
            completion = vllm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        return completion.choices[0].message.content
    
    elif provider == "sglang":
        assert "endpoint" in llm_config, "endpoint is required for sglang"
        sglang_client = openai.OpenAI(
            base_url=llm_config["endpoint"],
            api_key="None",
        )
        if response_format is not None:
            completion = sglang_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "foo",
                        "schema": response_format.model_json_schema(),
                    },
                }
            )
        else:
            completion = sglang_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        return completion.choices[0].message.content