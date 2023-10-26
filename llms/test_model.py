import torch
from create_prompt import create_lyrics_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def generate_response(model, tokenizer, prompt, temp=0.7, top_p=0.9, repetition=1.05):
    inputs = tokenizer(f'「{prompt}」を使って歌詞を作成してください。\n作った歌詞は、「', return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=repetition,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f'({temp=}, {top_p=}, {repetition=})')
    print(output)
    print()


def generate_response_engineered(model, tokenizer, prompt, topics, temp=0.7, top_p=0.9, repetition=1.05):
    inputs = tokenizer(f'「{prompt}」\nこれらの歌詞の例に基づいて、いきものがかりのスタイルで「{topics}」の言葉を使って、曲を作成して下さい。歌詞は「', return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=repetition,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(f'({temp=}, {top_p=}, {repetition=})')
    print(output)
    print()


def main():
    prompt = create_lyrics_prompt(3)
    model_name = 'cyberagent/open-calm-small'
    model_name_1b = 'cyberagent/open-calm-1b'

    models = [model_name, model_name_1b]
    # settings = [(0.7, 0.9, 1.05), (0.7, 0.8, 1.05), (0.7, 0.7, 1.05), (0.7, 0.9, 1.5), (0.7, 0.9, 20.0), (0.9, 0.9, 1.05), (0.8, 0.9, 1.05)]
    # settings = [(2.0, 0.9, 1.05), (2.0, 0.9, 10.0), (2.0, 0.9, 20.0), (2.0, 0.9, 40.0), (2.0, 0.9, 60.0), (5.0, 0.9, 10.0)]
    # settings = [(2.0, 0.9, 1.05), (1.0, 0.95, 10.0)]
    settings = [(0.7, 0.9, 1.05)]

    for model_type in models:
        print(f'RESULTS FOR {model_type:=^50}')
        for setting in settings:
            temperature, top_p, repetition = setting
            model = AutoModelForCausalLM.from_pretrained(model_type, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_type, local_files_only=True)
            generate_response_engineered(model, tokenizer, prompt, '雨', temperature, top_p, repetition)
            #雨、愛、夏休み

if __name__ == '__main__':
    main()
