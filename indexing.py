import argparse
import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import logging
import re
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Text Normalization
# -------------------------------
def normalize_string(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).lower()


# -------------------------------
# Inverted Index Search Engine
# -------------------------------
class SearchEngine:
    def __init__(self):
        self._index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._documents: dict[str, str] = {}

    def index(self, url: str, content: str) -> None:
        self._documents[url] = content
        words = normalize_string(content).split()
        for word in words:
            self._index[word][url] += 1

    def bulk_index(self, documents: list[tuple[str, str]]):
        for url, content in documents:
            self.index(url, content)

    def get_urls(self, keyword: str) -> dict[str, int]:
        keyword = normalize_string(keyword)
        return self._index[keyword]


# -------------------------------
# Web Content Cleaner
# -------------------------------
def clean_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)


# -------------------------------
# Web Crawling Logic
# -------------------------------
def parse_feed(feed_url):
    try:
        # If this was to be an RSS feed, you can add RSS parsing code here
        return [feed_url]
    except Exception as e:
        print(f"Error parsing feed {feed_url}: {e}")
        return []


async def fetch_content(session, url):
    async with session.get(url) as response:
        return await response.text()


async def process_feed(feed_url, session, loop):
    try:
        post_urls = await loop.run_in_executor(None, parse_feed, feed_url)
        tasks = [fetch_content(session, post_url) for post_url in post_urls]
        post_contents = await asyncio.gather(*tasks)
        cleaned_contents = [clean_content(content) for content in post_contents]
        return list(zip(post_urls, cleaned_contents))
    except Exception as e:
        print(f"Error processing feed {feed_url}: {e}")
        return []


# -------------------------------
# Argument Parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-path", help="Path to file with URLs to crawl", required=True)
    return parser.parse_args()


# -------------------------------
# Main Function for Crawling and Indexing
# -------------------------------
async def main(feed_file):
    search_engine = SearchEngine()

    # Reading the URLs from a file
    with open(feed_file, "r") as file:
        feed_urls = [line.strip() for line in file]

    async with aiohttp.ClientSession() as session:
        loop = asyncio.get_event_loop()
        tasks = [process_feed(feed_url, session, loop) for feed_url in feed_urls]
        results = await asyncio.gather(*tasks)

    # Flatten the results (list of tuples of URL and content)
    flattened_results = [item for sublist in results for item in sublist]

    # Save crawled content to a DataFrame (output as .parquet file)
    df = pd.DataFrame(flattened_results, columns=["URL", "content"])
    df.to_parquet("gymnastics_output.parquet", index=False)
    print("✅ Crawled and saved content to gymnastics_output.parquet")

    # Index the documents using the inverted index
    search_engine.bulk_index(flattened_results)
    print("🔍 Inverted index built successfully")

    # Allow for keyword searching
    query = input("Enter a keyword to search (e.g., 'vault', 'Simone', 'routine'): ")
    matches = search_engine.get_urls(query)

    if matches:
        print(f"\nResults for '{query}':")
        for url, count in matches.items():
            print(f"{url} ({count} times)")
    else:
        print(f"No documents found containing '{query}'.")


# -------------------------------
# Run the Script
# -------------------------------
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.feed_path))
