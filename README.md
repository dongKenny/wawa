和和 (wawa) - Japanese Lyrics Generation
============

## Project Description

和和 is a project to use various generative AIs (LLMs) to produce Japanese song lyrics.

The initial version of the project used only n-grams (NLTK's MLE model) to generate lyrics and a RoBERTa MLM to populate gaps in provided inputs. The n-grams were able to produce some output which improved with higher n, but the time needed to create/load them was becoming too high as n grew.

The intent of 和和 is to use LLMs created for Japanese and tuning them to create song lyrics instead. 

## How to use 和和

```
# From the project root

python3.9 lyrics/corpus_dataset.py # Run it to create the cleaned lyrics sets


python3.9 llms/deploy.py # Will deploy to localhost through ngrok if the NGROK_SECRET is set, delete these lines if you want to run it without that (Note: the RNN will raise an error even with setting GPU memory growth options so only Rinna will work)


python3.9 llms/wawa_rnn.py # Will prompt you for seed text to generate lyrics with the RNN
```


## Part 1 - Data Collection
 
I scraped 331,597 pages from https://www.uta-net.com/ to collect Japanese song lyrics. The `uta_net_scraper.py` script asynchronously retrieves all of the artists in 五十音順 order (a Japanese syllabry order), retrieves their list of songs, and scrapes the lyrics from there. The resulting `lyrics.txt` is arranged as `Artist SEP Title SEP Lyrics` (the literal use of "SEP" is to have an easy way to split everything). 

There were no cases of a `ServerDisconnectedError` from `aiohttp`, but if you do encounter any errors, it is easy to identify the point of failure in the log/lyrics and then create a separate script to pick up from there. For example, if it failed halfway down on one of the pages for an artist, the `scrape_artist_names()` function can be copied, adjusted to only grab that page index, split to only continue from the artists that were not collected, replace the names list, and finally be scraped and appended to your `lyrics.txt`. Once that single page is done, just adjust the start index for the range used in `scrape()` to be the next set of artists so the rest of the scraping can continue as normal.


## Part 2 - Training

### OpenCALM-7b
I trained various HuggingFace LLMs and found very inconsistent results. The first model I fine tuned was cyberagent's [OpenCALM-7b model](https://huggingface.co/cyberagent/open-calm-7b). I created a dataset by just feeding it artists, song titles, and song lyrics. The majority of the time when generating text from the model, I received sequences loosely related to the prompt keywords (e.g. 雨 rain) or repeated gibberish sequences with certain temperatures. On rare occasion, I was able to generate actual lyrics like (Google Translation because I have many lyrics that I do not want to translate manually):

```
「 見つめ合う瞳が 見つめ合う瞳が 雨に濡れたあなたの髪に 触れた瞬間 鮮やかに染められるだろう 雨に打たれながら うつむいて歩く あなたの横顔を見てたんだ 今でもあなたに会えて嬉しいよ そして今はただ一緒に過ごしたい」

"The eyes staring at each other, the eyes staring at each other, The moment I touch your rain-drenched hair, it will be dyed brightly. As you walk with your head down in the rain, I was looking at your profile. I'm still happy to see you. And now I just want to spend time with you."
```

```
「 冷たい雨に打たれながら ただ泣けずに 全てを溶かし 消していく 消えていく涙に ただ悲しくもなる 君が去った その日から 僕の心も冷えて 冷えて 凍えていく」

"While being hit by the cold rain, I just can't cry, I can't help but melt and erase everything, and the fading tears make her feel sad.Ever since the day you left, my heart has also become cold, cold, frozen."
```

### Rinna 

The second model I used was rinna, [japanese-gpt-neox-3.6b-instruction-ppo](https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-ppo). I chose it because it was an instruct model, so I could prompt it with some more instructions and context.

The prompt used was: "システムは日本の作詞家です。システムに歌の名前と歌詞の例をあげて、{artist}のスタイルで歌詞を作成して下さい。そのプロンプト形式は：「曲名」という歌を作って下さい。{artist}のスタイルで「話題の言葉」の言葉を使って、曲を作成して下さい。"

System is a Japanese lyricist. System will be provided song names and examples of lyrics, please create lyrics in the style of {artist}. The prompt format is: Please create a song called "Song Title". In the style of {artist}, using "Keywords", create a song.

Here is an example of lyrics using rain (in the style of いきものがかり, about rain):

```
「 降り続ける雨が世界を濡らす静かに染み渡る優しい光が心を照らす光 その愛に抱かれて 世界の隅々に輝きが生まれる それがこの今の気持ちに合わなくても 雨が降るたびに 私は強くなる 絶望的な苦しみに耐えながら そして その闇に消えゆく光 そして その儚くも美しい愛に その全てが愛されている」

"The rain that continues to fall wets the world. A gentle light that quietly permeates the heart. Embraced by that love, a glow is born in every corner of the world. Even if it doesn't match my current feelings, every time it rains, I... I grow stronger, endure the hopeless suffering, and the light disappears into the darkness, and I am loved by all of that fleeting yet beautiful love."
```

There are some issues that can arise from the small token window in the prompt. Since the model isn't fine tuned on all lyrics in the dataset, it can accept ~2000 tokens of lyrics (roughly enough for 3 songs). This can occasionally create some issues of plagiarism or minimal output with missing patterns.

This output just says the song title and keyword back without actually making any lyrics:
```
Artist: 結束バンド (Kessoku Band)

Title: ひとりぼっち (Loneliness)

Keyword: 眩しい (Dazzling)

システム: 「ひとりぼっち」の歌詞は「眩しい」の言葉を使って、曲を作成してください。

System: Please create a song using the word ``dazzling'' as the lyrics for ``Hitoribocchi''.
```

This output accidentally plagiarizes part of [星座になれたら](https://www.uta-net.com/song/330462/) from the same band, likely because it was used in the prompt:
```
Artist: 結束バンド (Kessoku Band)

Title: ギターと孤独と蒼い惑星 (Guitar, Loneliness, and Blue Planet)

Keyword: 私 (I)

システム: 「ギターと孤独と蒼い惑星」の歌詞は「ギターと孤独と蒼い惑星 ひとりぼっちいっぱいの音を聞いてきた回り続けて 幾千年も一瞬でもいいから 誰か心のノイズをとってわたしを覗いてよ誰も心の奥には入れないけれど期待してしまう そんな夜Distortion it's Motion始まったらもう止まらない 制限も経験で塗り替えられるさそうだろう ディストーション いつもそう左脳追い越して心が走り出す君の方へと次の音にエスコート日々の憂いにディストーション みんなから愛されて「いいや 僕は ずっと一人きりさ」君と集まって星座になれたら彗星みたい 流れるひとりごと消えていく 残像は 真夜中のプリズム君と集まって星座になれたら夜広げて 描こう絵空事暗闇を 照らすような つないだ線 解かないよ君がどんなに眩しくても」です。 

(つないだ線 解かないよ 君がどんなに眩しくても are taken)

System: The lyrics of "Guitar, Solitude, and Blue Planet" are "Guitar, Solitude, and Blue Planet. I've been listening to so many sounds all by myself. I've been spinning around for thousands of years, even if it's only for a moment. Someone please take away the noise in my heart and look at me. No one can go deep into my heart, but I'm looking forward to it On nights like that, Distortion It's Motion Once it starts, I can't stop It seems like the limitations can be rewritten with experience Distortion It's always like that, my left brain overtakes my heart and my heart starts running towards you. Escort to the sound Distortion to the daily sadness Being loved by everyone ``No, I'll always be alone'' If I could gather with you and become a constellation, it would be like a comet The afterimage disappears with each passing monologue The afterimage is a midnight prism Let's gather together with you and become a constellation Let's spread out at night and draw a picture of connected lines that illuminate the darkness. No matter how dazzling you are, I won't be able to unravel them."
```
### TF RNN

The final model I used was just a simple RNN fed with pure lyrics. Since I had a primary goal of consistently generating lyrics, regardless of quality, I decided to try an RNN to see how it would try to match patterns/structure/vocabulary used in songs. Although it can't be as targeted in terms of titling/keyword relation, here are some results after only 10 epochs:

```
声が響く手を透かしてる誰にでもいい人生が生まれるのを待ちどう生きる月にまぶしさおとが何も持たずこの宇宙に昇るかけら集う理想とゆらめいたこの瞬間を手に入れる為に死に到来思い知らされたい何が為になんとたびれ今すぐ一人で逝く震える体の底に本当のところの出逢いのなかで飛ぶ羽を広げ大地広がるわけがないけれど全てを失くした日まで生きてゆけやりたなみたくなる意味をもらおうとしろ

A voice echoes through the hand, waiting for a good life to be born to everyone. How do you live? A glare on the moon rises to this universe with nothing. Death has arrived in order to obtain the ideals and this flickering moment where the fragments gather. I want to be reminded of what and why I'm so tired. I'm going to die alone right now. At the bottom of my trembling body, I'm going to fly in the midst of a true encounter. There's no way I can spread my wings and spread out across the land. But I'll keep living until the day I lose everything. Try to find a meaning that makes you want to read it

 心の奥には　高く飛び立て誰かが向かう自分次第で生きていくやったつも眠れなかった毎日が地元の穴何もかも打ち歩が射す術たとえ孤独にされし者には誰も気がつけば寒い夜もあるけどもう一度　人は生きてゆける　惜しみなくこの星に生まれてたら涙が流れた輝きを手に入れてたからその日を踏み出してプライドは捨ててもクルクルと近づいて来るライブ一人になるから先の光を追い抜いてたな明日はまだ2人があるけど　まだ平等にしたいし笑う金かけてきて僕らは今日もあの物語　続きを止めずプライドと悲しみが積もるクラクラして生きて耐えてきたその音　聞こえる何かに　生きてるみんながいた　流した涙は流れ落ちるあたたかさを指で作った本当の手を握って生きて行けますか生き方を　決して幾度も進んで生

  In the back of my heart, I'm flying high and someone is heading towards me. I live my life depending on who I am, but every day I couldn't sleep was a hole in my hometown. Even if I was alone, there would be cold nights without anyone realizing it. Once again, people can live. If I had been born on this planet generously, I would have had the shine that made tears flow, so I'll step out on that day. Even if I throw away my pride, I'll come closer and closer. I'll be alone, so I'll overtake the light ahead. Tomorrow there are still two of us, but we still want to make them equal, and we still have to spend money to laugh. Today, too, we continue that story without stopping. Our pride and sadness are piling up. We live in a state of dizziness, and we're living in something that we can hear, the sounds that we endured. The tears that I shed were all flowing down. Can I hold the real hand that I made with my fingers and live my life?
```

## Part 3 - Future Work

The instruct model and RNN have the most promise. The instruct model is extremely sensitive to minor adjustments in the prompt and could be further optimized in that area, but the more long-term and effective gains to be had would be from fine tuning the model. Since I did not find much documentation about how to best fine tune the instruct model (without any RLHF), I decided not to waste time with long training times for little gain like I had with the OpenCALM model. 

The RNN is also an interesting direction since it could be easily modified to incorporate more structure and focused keyword direction, and with more epochs and hyperparameter tuning shows promise and quick improvement. Keywords could be randomly incorporated during the one-shot text generation, for example.

The goal of this project is not to replace any artist or be used for any monetary gain; it is to simply explore Japanese NLG to try matching a more lyrical style and structure to see if it is possible to maintain a more humanlike result. As such, music/vocal synthesis is not in the intended project direction (especially since artists' performances are more easily monetized if able to be styled to any lyrics + no permission from them).
