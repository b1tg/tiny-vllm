import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
  # Get the absolute path to the model
  script_dir = os.path.dirname(os.path.abspath(__file__))
  path = os.path.join(script_dir, "..", "Qwen3-0.6B")

  # Fallback to ~/huggingface if the local path doesn't exist
  if not os.path.exists(path):
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

  if not os.path.exists(path):
    raise FileNotFoundError(f"Model not found. Please place Qwen3-0.6B at {path}")

  tokenizer = AutoTokenizer.from_pretrained(path)
  llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

  sampling_params = SamplingParams(temperature=0.6, max_tokens=10)  # Reduced for testing
  prompts = [
    "Hello",  # Shorter prompt for testing
  ]
  prompts = [
    tokenizer.apply_chat_template(
      [{"role": "user", "content": prompt}],
      tokenize=False,
      add_generation_prompt=True,
    )
    for prompt in prompts
  ]
  outputs = llm.generate(prompts, sampling_params)

  for prompt, output in zip(prompts, outputs):
    print("\n")
    print(f"Prompt: {prompt!r}")
    print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
  main()
