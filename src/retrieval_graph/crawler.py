import os
import uuid
from playwright.async_api import async_playwright
from urllib.parse import urldefrag, urljoin, urlparse

class WebCrawler:
    def __init__(self, starter_urls, hops, allowed_domains, storage_folder):
        self.starter_urls = starter_urls
        self.hops = hops
        self.allowed_domains = allowed_domains
        self.storage_folder = storage_folder
        self.visited_urls = set()
        self.crawled_pages = []

        # Ensure the storage folder exists
        os.makedirs(self.storage_folder, exist_ok=True)

    def is_allowed(self, url):
        domain = urlparse(url).netloc
        return any(domain.endswith(allowed) for allowed in self.allowed_domains)

    def normalize_url(self, url):
        # Remove fragment
        url, _ = urldefrag(url)

        if url.count('/') == 2 and not url.endswith('/'):
            url += '/'
        else:
            # Remove trailing slash for other cases
            url = url.rstrip('/')

        return url

    def save_page_content(self, content, url):
        file_name = f"{uuid.uuid4().hex}.html"
        file_path = os.path.join(self.storage_folder, file_name)

        # Save the content to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        # Track the crawled page
        self.crawled_pages.append({
            "url": url,
            "local_filepath": file_path,
            "size": len(content)
        })

    async def crawl(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()

            queue = [(self.normalize_url(url), 0) for url in self.starter_urls]

            while queue:
                current_url, depth = queue.pop(0)
                normalized_url = self.normalize_url(current_url)

                if normalized_url in self.visited_urls or depth > self.hops:
                    continue

                print(f"Crawling: {current_url}")
                try:
                    page = await context.new_page()
                    response = await page.goto(current_url, timeout=10000)
                    if response.status < 200 or response.status >= 400:
                        print(f"Failed to crawl {current_url}: {response.status}")
                        continue

                    self.visited_urls.add(normalized_url)

                    # Save the content of the visited page
                    content = await page.content()
                    self.save_page_content(content, current_url)

                    # Extract links
                    links = await page.locator("a[href]").element_handles()
                    for link in links:
                        href = await link.get_attribute("href")
                        if href:
                            normalized_href = self.normalize_url(urljoin(current_url, href))

                            if self.is_allowed(normalized_href) and normalized_href not in self.visited_urls:
                                queue.append((normalized_href, depth + 1))

                    await page.close()

                except Exception as e:
                    print(f"Failed to crawl {current_url}: {e}")

            await browser.close()