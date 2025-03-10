import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model_path = "qwen_finetuned"  # Ensure this folder is included in submission

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

def generate_answer(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    question = "How does DeepSeek-V3 optimize training?"
    print(generate_answer(question))
