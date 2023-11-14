import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, \
    TrainingArguments
from create_prompt import create_rinna_prompt


torch.cuda.empty_cache()


def load_dataset():
    return load_from_disk('/raid/kdong4/wawa/uta_dataset_instruction.hf')


def create_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    return AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", quantization_config=bnb_config)


def create_tokenizer():
    return AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-ppo", use_fast=False)


def create_data_collator(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def create_training_args():
    return TrainingArguments(
        output_dir='rinna_instruction_ppo/',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        remove_unused_columns=False,
        save_total_limit=2
    )


def generate_response(model, tokenizer, token_ids):
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


def main():
    tokenizer = create_tokenizer()
    model = create_model()
    
    prompt = create_rinna_prompt(4)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    generate_response(model, tokenizer, token_ids)


if __name__ == '__main__':
    main()
