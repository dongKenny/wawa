import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model, tokenizer, prompt='AIによって私達の暮らしは、'):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(output)


def main():
    model_path = ''

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(f'{model_path}', local_files_only=True)
    generate_response(model, tokenizer, '雨は、')


if __name__ == '__main__':
    main()