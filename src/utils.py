import torch
from config import settings

def create_call_qwen(model, tokenizer):
    """Return a function that calls Qwen with a prompt."""
    def call_qwen(prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        if temperature is None:
            temperature = settings.TEMPERATURE
        if max_tokens is None:
            max_tokens = settings.MAX_TOKENS

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=settings.TOP_P,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
    return call_qwen