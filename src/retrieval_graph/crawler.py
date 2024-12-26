import os
import uuid
from urllib.parse import urldefrag, urljoin, urlparse

from playwright.async_api import async_playwright

class WebCrawler:
    """
    An asynchronous web crawler that collects links starting from one or more
    seed URLs, subject to a specified maximum depth (number of hops) and
    restricted to certain allowed domains. Page contents are saved locally in
    a specified folder.

    Attributes:
        starter_urls (list[str]): The initial list of URLs from which to start crawling.
        hops (int): The maximum depth to crawl from the starter URLs.
        allowed_domains (list[str]): A list of domain names (or domain suffixes)
            that the crawler is allowed to follow.
        storage_folder (str): The folder where the crawler should store retrieved
            page contents.
        visited_urls (set[str]): A set of all URLs that have been visited (and
            hence will not be visited again).
        crawled_pages (list[dict]): A list of dictionaries containing metadata for each
            crawled page (e.g., URL, local file path, size).
    """

    def __init__(self, starter_urls, hops, allowed_domains, storage_folder):
        """
        Initializes the WebCrawler with starter URLs, maximum hops, allowed domains,
        and a folder for storing page contents.

        Args:
            starter_urls (list[str]): The initial URLs to start the crawl from.
            hops (int): The maximum number of hops/depth from the starter URLs.
            allowed_domains (list[str]): The domains or domain endings allowed for
                following links.
            storage_folder (str): The path to the folder where HTML files are stored.
        """
        self.starter_urls = starter_urls
        self.hops = hops
        self.allowed_domains = allowed_domains
        self.storage_folder = storage_folder
        self.visited_urls = set()
        self.crawled_pages = []

        # Ensure the storage folder exists
        os.makedirs(self.storage_folder, exist_ok=True)

    def is_allowed(self, url):
        """
        Checks if a URL is allowed based on the domains the crawler can visit.

        Args:
            url (str): The URL whose domain is to be checked.

        Returns:
            bool: True if the domain of the URL is in the list of allowed domains,
            False otherwise.
        """
        domain = urlparse(url).netloc
        return any(domain.endswith(allowed) for allowed in self.allowed_domains)

    def normalize_url(self, url):
        """
        Normalizes a URL by removing URL fragments and ensuring consistency in trailing
        slashes to avoid duplicates (e.g., http://example.com vs. http://example.com/).

        - If the URL has exactly two slashes (e.g., scheme://domain) and no trailing slash,
          it adds the trailing slash.
        - Otherwise, it strips trailing slashes for consistency.

        Args:
            url (str): The URL to normalize.

        Returns:
            str: The normalized URL (fragment removed, trailing slash handled).
        """
        # Remove fragment
        url, _ = urldefrag(url)

        if url.count('/') == 2 and not url.endswith('/'):
            # If the URL is something like 'http://example.com', append '/'
            url += '/'
        else:
            # Otherwise, remove trailing slash
            url = url.rstrip('/')

        return url

    def save_page_content(self, content, url):
        """
        Saves the HTML content of a visited page to a local file, and records
        metadata in crawled_pages.

        Args:
            content (str): The HTML source code of the crawled page.
            url (str): The URL of the page whose content is being saved.
        """
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
        """
        Performs the asynchronous crawling process. It launches a headless Chromium
        browser using Playwright, initializes a queue of URLs to visit, and iterates
        over them up to the allowed hop depth. For each visited page:

        - Loads the page in a new browser tab.
        - Saves its content locally.
        - Extracts and normalizes all links from <a href="..."> elements.
        - Enqueues any links to allowed domains that haven't been visited.

        This method should be awaited to ensure the event loop completes the crawl.

        Raises:
            Exception: Logs and skips any pages that fail to crawl.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()

            # Initialize the queue with (URL, depth) tuples
            queue = [(self.normalize_url(url), 0) for url in self.starter_urls]

            # While there are still URLs to crawl...
            while queue:
                current_url, depth = queue.pop(0)
                normalized_url = self.normalize_url(current_url)

                # Skip if already visited or if the depth exceeds the maximum hops
                if normalized_url in self.visited_urls or depth > self.hops:
                    continue

                print(f"Crawling: {current_url}")
                try:
                    page = await context.new_page()
                    response = await page.goto(current_url, timeout=10000)
                    
                    # Check for valid response status
                    if response.status < 200 or response.status >= 400:
                        print(f"Failed to crawl {current_url}: {response.status}")
                        await page.close()
                        continue

                    # Mark URL as visited
                    self.visited_urls.add(normalized_url)

                    # Save the content of the visited page
                    content = await page.content()
                    self.save_page_content(content, current_url)

                    # Extract and process links
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
