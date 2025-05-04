import argparse
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
import json
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncRecursiveCrawler:
    def __init__(self, seed_urls, max_pages=100, max_depth=2, num_workers=4):
        self.url_queue = asyncio.Queue()
        for url in seed_urls:
            self.url_queue.put_nowait((url, 0))  # (url, depth)
        self.visited = set()
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.num_workers = num_workers
        self.results = []
        self.running = True
        self.session = None

    async def fetch(self, url):
        if not self.running or not self.session:
            return ""

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        for attempt in range(3):
            try:
                async with self.session.get(url, headers=headers, timeout=15) as response:
                    if response.status == 200 and "text/html" in response.headers.get("content-type", ""):
                        await asyncio.sleep(1)  # polite delay!
                        return await response.text()
                    else:
                        logger.warning(f"Non-200 status for {url}: {response.status}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                await asyncio.sleep(2 * (attempt + 1))  # exponential backoff
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

    async def worker(self, worker_id):
        try:
            while self.running and len(self.visited) < self.max_pages:
                try:
                    url, depth = await asyncio.wait_for(self.url_queue.get(), timeout=5)
                except asyncio.TimeoutError:
                    # No more URLs in queue, check if we should exit
                    if self.url_queue.empty():
                        logger.info(f"Worker {worker_id} found empty queue, exiting")
                        break
                    continue

                if url in self.visited or depth > self.max_depth:
                    self.url_queue.task_done()
                    continue

                self.visited.add(url)
                html = await self.fetch(url)

                if not html:
                    self.url_queue.task_done()
                    continue

                content = self.clean_content(html)
                links = self.extract_links(html, url)
                self.results.append({"url": url, "content": content, "links": links, "depth": depth})

                # Only add new URLs if we're still below our limit
                if len(self.visited) < self.max_pages:
                    for link in links:
                        if link not in self.visited:
                            await self.url_queue.put((link, depth + 1))

                self.url_queue.task_done()

                if len(self.visited) % 10 == 0:
                    logger.info(f"Crawled {len(self.visited)} pages...")
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
        finally:
            logger.info(f"Worker {worker_id} shutting down")

    async def crawl(self):
        # Create a single session for all workers
        self.session = aiohttp.ClientSession()
        try:
            # Start workers
            workers = [asyncio.create_task(self.worker(i)) for i in range(self.num_workers)]

            # Wait for either max pages or queue empty
            while len(self.visited) < self.max_pages and not self.url_queue.empty():
                await asyncio.sleep(1)

                # Check if we've reached our goal
                if len(self.visited) >= self.max_pages:
                    logger.info(f"Reached max pages limit: {self.max_pages}")
                    break

            # Signal workers to stop
            self.running = False

            # Wait for queue to drain with timeout
            try:
                await asyncio.wait_for(self.url_queue.join(), timeout=10)
                logger.info("Queue successfully drained")
            except asyncio.TimeoutError:
                logger.warning("Queue drain timed out")

            # Cancel all workers
            for w in workers:
                w.cancel()

            # Wait for workers to finish cancellation
            await asyncio.gather(*workers, return_exceptions=True)

        finally:
            # Always close the session
            await self.session.close()
            self.session = None
            logger.info(f"Crawl completed with {len(self.visited)} pages")

async def main():
    args = parse_args()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    with open(args.feed_path, "r") as f:
        seed_urls = [line.strip() for line in f if line.strip()]
    print("Length of seed urls:", len(seed_urls))

    crawler = AsyncRecursiveCrawler(
        seed_urls=seed_urls,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        num_workers=10
    )

    try:
        await crawler.crawl()
    except asyncio.CancelledError:
        logger.info("Main task cancelled")

    logger.info(f"Crawled {len(crawler.visited)} unique pages.")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(crawler.results, f, ensure_ascii=False, indent=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-path", required=True, help="Path to feeds.txt")
    parser.add_argument("--max-pages", type=int, default=100000)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--output", default="output.json")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())