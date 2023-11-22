import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from create_prompt import create_rinna_prompt
from lyrics.lyric_options import map_artists_to_songs_and_lyrics, create_artist_datalists


torch.cuda.empty_cache()


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


def generate_with_pipeline(model, tokenizer, artist, song_title, song_keyword, song_data):
    with torch.autocast("cuda"): 
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        seqs = pipe(create_rinna_prompt(3, artist, song_title, song_keyword, song_data, True), max_new_tokens=512)
        output = seqs[0]['generated_text'].split('<NL>')[-1]
        return output


def main():
    tokenizer = create_tokenizer()
    model = create_model()
    print(generate_with_pipeline(model, tokenizer, 'いきものがかり', '夏の雨', '雨', map_artists_to_songs_and_lyrics()))


if __name__ == '__main__':
    main()
