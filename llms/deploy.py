import os
import time
from dotenv import load_dotenv

import tensorflow as tf

from rinna_instruction import create_model, create_tokenizer, generate_with_pipeline
from lyrics.lyric_options import map_artists_to_songs_and_lyrics, create_artist_datalists
from llms.wawa_rnn import rnn_generate_text

from pyngrok import ngrok
from flask import Flask, render_template, request

# Load model once
model, tokenizer = create_model(), create_tokenizer()
load_dotenv()

app = Flask(__name__)
app.config.from_mapping(
    BASE_URL='http://localhost:5000'
)

# Create an Ngrok tunnel to localhost
ngrok.set_auth_token(os.getenv('NGROK_SECRET'))
public_url = ngrok.connect(5000).public_url
print(f'* ngrok tunnel "{public_url}" -> http://127.0.0.1:5000')

n_grams = dict()
initialized = False

http_tunnel = ngrok.connect()
artist_data = map_artists_to_songs_and_lyrics()
datalist_options = '\n'.join(create_artist_datalists(artist_data))


@app.route("/")
def index():
    global initialized

    # First load of / page, loads in the n_gram models and creates the pipeline for the fill-mask
    if not initialized:
        # Set to True immediately to prevent multiple tabs from loading models
        initialized = True

    return render_template("index.html", title="Index")


@app.route("/rinna", methods=["GET", "POST"])
def rinna():
    artist = request.form.get("artist", "")
    title = request.form.get("title", "")
    keyword = request.form.get("keyword", "")
    lyrics = 'No lyrics generated'

    if artist and title and keyword:
        lyrics = generate_with_pipeline(model, tokenizer, artist, title, keyword, artist_data)

    return render_template("rinna.html", artist=artist, title=title, keyword=keyword, jp_lyrics=lyrics, artist_options=datalist_options, action="/rinna")


@app.route("/rnn", methods=["GET", "POST"])
def rnn():
    keyword = request.form.get("keyword", "")
    lyrics = 'No lyrics generated'

    if keyword:
        # Error with loading both models on the GPUs, can just run wawa_rnn with a keyword to generate
        lyrics = rnn_generate_text(keyword)

    return render_template("rnn.html", keyword=keyword, jp_lyrics=lyrics, action="/rnn")


if __name__ == "__main__":
    app.run(debug=False)
