from more_itertools import chunked, flatten
from aiohttp import TCPConnector
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import logging

UTA_NET_BASE_URL = 'https://www.uta-net.com'
logging.basicConfig(filename='lyrics.log', level=logging.INFO)


async def scrape_lyrics(session, songs, writer):
    lyrics = []

    # Aync gathers the lyrics
    pages = [asyncio.create_task(session.get(f'{UTA_NET_BASE_URL}{song[0]}')) for song in songs]
    page_responses = await asyncio.gather(*pages)

    # Scrape the lyrics
    for page in page_responses:
        soup = BeautifulSoup(await page.text(), 'html.parser')
        lyric = soup.find("div", {"id": "kashi_area"})
        if lyric:
            lyrics.append(lyric.text)

    for song, lyric in zip(songs, lyrics):
        logging.info(f'{song[2]} - {song[1]}')
        writer.write(f'{song[2]} SEP {song[1]} SEP {lyric}\n')


async def scrape_artist_songs(session, names):
    songs = []

    # Aync gather the artists' song hrefs and then scrape the lyrics
    pages = [asyncio.create_task(session.get(f'{UTA_NET_BASE_URL}{name}')) for name in names]
    page_responses = await asyncio.gather(*pages)

    # Parse the artist pages to retrieve song hrefs and then scrape the lyrics
    for page in page_responses:
        soup = BeautifulSoup(await page.text(), 'html.parser')
        song_names = [song_list.find_all('a') for song_list in soup.find_all('tbody', class_='songlist-table-body')]
        artist = soup.find('h2', class_='my-2 my-lg-0 mx-2').text.split('の歌詞')[0].lstrip()

        for song_name in song_names:
            for song in song_name:
                if 'song' in song['href']:
                    title = song.find('span', class_='fw-bold songlist-title pb-1 pb-lg-0').string
                    songs.append((song['href'], title, artist))

    return songs


async def scrape_artist_names(session, page_index):
    names = []

    # Aync gather the artist names and hrefs to the pages containing their song list to further scrape from
    pages = [asyncio.create_task(session.get(f'{UTA_NET_BASE_URL}/name_list/{page_index}'))]
    page_responses = await asyncio.gather(*pages)

    # Parse the lists (grouped by first and second character order), parse sublists for artist name/hrefs
    for page in page_responses:
        soup = BeautifulSoup(await page.text(), 'html.parser')
        name_lists = [name_list.find_all('a') for name_list in soup.find_all('dl', class_='artist-list')]

        for name_list in name_lists:
            names.append([name['href'] for name in name_list if name.string != 'アルバム'])

    return list(flatten(names))


async def scrape(session, writer):
    # character index for ん is 70 in their pagination, it was scraped beforehand and then the rest were run
    char_index = 45  # 五十音順 character index goes 0-44 on their pagination

    for i in range(char_index):
        try:
            names = await scrape_artist_names(session, i)

            songs = []
            for name_split in list(chunked(names, 50)):
                songs.append(await scrape_artist_songs(session, name_split))
            songs = list(flatten(songs))

            for song_split in list(chunked(songs, 50)):
                await scrape_lyrics(session, song_split, writer)
        except aiohttp.ServerDisconnectedError:
            logging.error(f'Failed for index {i}')


async def main():
    async with aiohttp.ClientSession(connector=TCPConnector(force_close=True)) as session:
        with open('lyrics.txt', "a", newline="", encoding="UTF-8") as writer:
            await scrape(session, writer)


if __name__ == '__main__':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.get_event_loop().run_until_complete(main())
