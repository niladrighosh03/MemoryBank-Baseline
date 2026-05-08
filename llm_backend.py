import os

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OPENAI_MODEL = "zai-org/GLM-5.1-FP8"
DEFAULT_OPENAI_BASE_URL = "https://api.vultrinference.com/v1"
DEFAULT_OPENAI_TEMPERATURE = 0.1


class HuggingFaceChatBackend:
    def __init__(self, model_name: str):
        print(f"Loading local HuggingFace chat model: {model_name} ...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        print("  Local model loaded.\n")

    def generate(self, system_msg: str, user_msg: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        new_tokens = output_ids[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAICompatibleChatBackend:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        temperature: float = DEFAULT_OPENAI_TEMPERATURE,
        enable_thinking: bool = False,
    ):
        if not api_key:
            raise ValueError(
                "Missing API key for OpenAI-compatible backend. "
                "Set OPENAI_API_KEY or pass --api_key."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"Using OpenAI-compatible endpoint: {base_url}")
        print(f"  Remote model: {model_name}\n")

    def generate(self, system_msg: str, user_msg: str, max_new_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=self.temperature,
            max_tokens=max_new_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            },
        )
        return (response.choices[0].message.content or "").strip()


def load_chat_backend(
    backend: str = "qwen",
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = DEFAULT_OPENAI_TEMPERATURE,
    enable_thinking: bool = False,
):
    backend = backend.lower()

    if backend == "qwen":
        return HuggingFaceChatBackend(model_name or DEFAULT_QWEN_MODEL)

    if backend == "openai":
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        return OpenAICompatibleChatBackend(
            model_name=model_name or DEFAULT_OPENAI_MODEL,
            api_key=resolved_api_key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            temperature=temperature,
            enable_thinking=enable_thinking,
        )

    raise ValueError(f"Unsupported backend: {backend}")
