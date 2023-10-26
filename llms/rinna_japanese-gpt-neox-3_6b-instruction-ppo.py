import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from create_prompt import create_rinna_prompt


torch.cuda.empty_cache()


def main():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", quantization_config=bnb_config)
    
    prompt = create_rinna_prompt(4)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    for i in range(2):
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                do_sample=True,
                max_new_tokens=512,
                temperature=0.7 + 0.1 * i,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
        output = output.replace("<NL>", "\n")
        print(output)


if __name__ == '__main__':
    main()