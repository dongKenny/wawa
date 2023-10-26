import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from lyrics.corpus_dataset import create_uta_dataset, tokenize_uta_dataset, uta_data_generator


torch.cuda.empty_cache() 


def load_dataset():
    return load_from_disk('/raid/kdong4/wawa/uta_dataset.hf')


def create_model():
    return AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-1b", device_map="auto", offload_folder="offload",
                                                torch_dtype=torch.float16)


def create_tokenizer():
    return AutoTokenizer.from_pretrained("cyberagent/open-calm-1b", model_max_length=512)


def create_data_collator(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def create_training_args():
    return TrainingArguments(
        output_dir='open_calm_1b/',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
        remove_unused_columns=False,
        save_total_limit=2
    )


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
    model = create_model()
    tokenizer = create_tokenizer()
    uta_data = load_dataset().train_test_split(test_size=0.2)
    tokenized_uta_data = uta_data.map(tokenize_uta_dataset, batched=True, fn_kwargs={'tokenizer': tokenizer}).remove_columns(uta_data['train'].column_names)
    data_collator = create_data_collator(tokenizer)
    training_args = create_training_args()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_uta_data['train'],
        eval_dataset=tokenized_uta_data['test'],
        data_collator=data_collator
    )

    trainer.train()


if __name__ == '__main__':
    main()
