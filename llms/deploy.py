import os
import time
from dotenv import load_dotenv

from rinna_instruction import create_model, create_tokenizer, generate_with_pipeline
from lyrics.lyric_options import map_artists_to_songs_and_lyrics, create_artist_datalists

from pyngrok import ngrok
from flask import Flask, render_template, request


model, tokenizer = create_model(), create_tokenizer()
load_dotenv()

app = Flask(__name__)
app.config.from_mapping(
    BASE_URL='http://localhost:5000'
)

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


if __name__ == "__main__":
    app.run(debug=False)
