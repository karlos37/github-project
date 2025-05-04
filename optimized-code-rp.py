from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

import argparse
import aiohttp
import asyncio
import pandas as pd
from bs4 import BeautifulSoup
import logging
import re
from collections import defaultdict
from math import log

def normalize_strings(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text).lower()

class SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75, num_clusters: int = 5):
        self._index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._documents: dict[str, str] = {}
        self.k1 = k1
        self.b = b
        self.num_clusters = num_clusters
        self.query_count_relevance = 0  # Track queries for relevance models
        self.query_count_expansion = 0  # Track queries for query expansion models

    @property
    def posts(self) -> list[str]:
        return list(self._documents.keys())

    @property
    def number_of_documents(self) -> int:
        return len(self._documents)

    @property
    def avdl(self) -> float:
        return sum(len(d) for d in self._documents.values()) / len(self._documents)
    
    

    def index(self, url: str, content: str) -> None:
        self._documents[url] = content
        words = normalize_strings(content).split()
        for word in words:
            self._index[word][url] += 1

    def bulk_index(self, documents: list[tuple[str, str]]):
        for url, content in documents:
            self.index(url, content)

    def get_urls(self, keyword: str) -> dict[str, int]:
        keyword = normalize_strings(keyword)
        return self._index[keyword]

    def idf(self, kw: str) -> float:
        N = self.number_of_documents
        n_kw = len(self.get_urls(kw))
        return log((N - n_kw + 0.5) / (n_kw + 0.5) + 1)

    def bm25(self, kw: str) -> dict[str, float]:
        result = {}
        idf_score = self.idf(kw)
        avdl = self.avdl
        for url, freq in self.get_urls(kw).items():
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * len(self._documents[url]) / avdl)
            result[url] = idf_score * numerator / denominator
        return result

    def search(self, query: str, use_expansion: bool = False) -> dict[str, float]:
        if use_expansion:
            self.query_count_expansion += 1
            query = self.expand_query(query)
        else:
            self.query_count_relevance += 1

        keywords = normalize_strings(query).split(" ")
        url_scores: dict[str, float] = {}
        for kw in keywords:
            kw_urls_score = self.bm25(kw)
            url_scores = self.update_url_scores(url_scores, kw_urls_score)
        return url_scores

    def update_url_scores(self, url_scores: dict[str, float], kw_urls_score: dict[str, float]) -> dict[str, float]:
        for url, score in kw_urls_score.items():
            if url in url_scores:
                url_scores[url] += score
            else:
                url_scores[url] = score
        return url_scores

    def expand_query(self, query: str) -> str:
        # Dummy query expansion: Add synonyms (in practice, use a thesaurus or model)
        expanded_query = query + " additional terms"
        return expanded_query

    def cluster_documents(self):
        # Use TF-IDF vectorizer to create feature vectors for the documents
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self._documents.values())

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)

        document_clusters = defaultdict(list)
        for idx, cluster in enumerate(clusters):
            document_clusters[cluster].append(list(self._documents.keys())[idx])

        return document_clusters

    def integrate_link_analysis(self):
        # Simple link analysis: You can add basic PageRank or a simplified version here
        # Here we simply assume that the URL is related to the score (basic assumption)
        # In reality, you'd integrate a more robust link-based algorithm like PageRank

        link_scores = defaultdict(float)
        for url in self._documents:
            # Simplified link analysis based on document length (can be more sophisticated)
            link_scores[url] = len(self._documents[url])
        return link_scores


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

    # Cluster documents to improve relevance
    clusters = search_engine.cluster_documents()
    print(f"📊 Number of clusters derived: {len(clusters)}")

    # Integrate link analysis into relevance models
    link_scores = search_engine.integrate_link_analysis()
    print("🔗 Link analysis integration complete")

    # Allow for keyword searching
    query = input("Enter a keyword to search (e.g., 'vault', 'Simone', 'routine'): ")
    use_expansion = input("Do you want to use query expansion (yes/no)? ").strip().lower() == "yes"
    matches = search_engine.search(query, use_expansion)

    if matches:
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        print(f"\nResults for '{query}':")
        for url, score in sorted_matches:
            print(f"{url} (score: {score})")
    else:
        print(f"No documents found containing '{query}'.")

# -------------------------------
# Run the Script
# -------------------------------
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.feed_path))
