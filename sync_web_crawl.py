import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import json

class SyncCrawler:
    def __init__(self, seed_urls, max_pages=100, max_depth=2):
        self.url_queue = [(url, 0) for url in seed_urls]  # (url, depth)
        self.visited = set()
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.results = []

    def fetch(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 ..."
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200 and (
                    "text/html" in resp.headers.get("content-type", "") or
                    "application/rss+xml" in resp.headers.get("content-type", "") or
                    "application/xml" in resp.headers.get("content-type", "")
            ):
                time.sleep(1)  # polite delay
                return resp.text
            else:
                print(f"Non-200 status for {url}: {resp.status_code}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return ""

    def extract_links(self, html, base_url):
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            link = urljoin(base_url, a["href"])
            parsed = urlparse(link)
            if parsed.scheme in ("http", "https"):
                links.append(link)
        return links

    def clean_content(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator=" ", strip=True)
        return text

    def crawl(self):
        while self.url_queue and len(self.visited) < self.max_pages:
            url, depth = self.url_queue.pop(0)
            if url in self.visited or depth > self.max_depth:
                continue
            self.visited.add(url)
            html = self.fetch(url)
            if not html:
                continue
            content = self.clean_content(html)
            links = self.extract_links(html, url)
            self.results.append({"url": url, "content": content, "links": links, "depth": depth})
            for link in links:
                if link not in self.visited:
                    self.url_queue.append((link, depth + 1))
            if len(self.visited) % 10 == 0:
                print(f"Crawled {len(self.visited)} pages...")

if __name__ == "__main__":
    with open("feeds.txt") as f:
        seed_urls = [line.strip() for line in f if line.strip()]
    crawler = SyncCrawler(seed_urls, max_pages=1000, max_depth=2)
    crawler.crawl()
    print(f"Crawled {len(crawler.visited)} unique pages.")
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(crawler.results, f, ensure_ascii=False, indent=2)
