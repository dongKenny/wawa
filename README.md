和和 (wawa) - Japanese Lyrics Generation
============

## Project Description

和和 is a project to use various generative AIs (LLMs) to produce Japanese song lyrics.

The initial version of the project used only n-grams (NLTK's MLE model) to generate lyrics and a RoBERTa MLM to populate gaps in provided inputs. The n-grams were able to produce some output which improved with higher n, but the time needed to create/load them was becoming too high as n grew.

The intent of 和和 is to use LLMs created for Japanese and tuning them to create song lyrics instead. 

## Part 1 - Data Collection
 
I scraped 331,597 pages from https://www.uta-net.com/ to collect Japanese song lyrics. The `uta_net_scraper.py` script asynchronously retrieves all of the artists in 五十音順 order (a Japanese syllabry order), retrieves their list of songs, and scrapes the lyrics from there. The resulting `lyrics.txt` is arranged as `Artist SEP Title SEP Lyrics` (the literal use of "SEP" is to have an easy way to split everything). 

There were no cases of a `ServerDisconnectedError` from `aiohttp`, but if you do encounter any errors, it is easy to identify the point of failure in the log/lyrics and then create a separate script to pick up from there. For example, if it failed halfway down on one of the pages for an artist, the `scrape_artist_names()` function can be copied, adjusted to only grab that page index, split to only continue from the artists that were not collected, replace the names list, and finally be scraped and appended to your `lyrics.txt`. Once that single page is done, just adjust the start index for the range used in `scrape()` to be the next set of artists so the rest of the scraping can continue as normal.
