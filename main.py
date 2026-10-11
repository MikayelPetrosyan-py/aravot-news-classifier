import logging
import re
import time
from urllib.parse import urljoin

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# SETTINGS
# ============================================================

ARTICLES_PER_CATEGORY = 100

CATEGORIES = {
    "politics": "https://www.aravot.am/category/news/politics/",
    "culture": "https://www.aravot.am/category/news/culture/",
    "sport": "https://www.aravot.am/category/news/sport/",
}

OUTPUT_FILE = "aravot_news.csv"
LOG_FILE = "scraper.log"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# SELENIUM
# ============================================================

options = Options()

# Հիմա browser-ը թող բացվի, որ տեսնենք ինչ է անում
# Եթե հետո ուզենք invisible աշխատացնել, կավելացնենք:
# options.add_argument("--headless")

options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

wait = WebDriverWait(driver, 15)


# ============================================================
# HELPERS
# ============================================================

def is_article_url(url):
    """
    Ստուգում է՝ URL-ը Aravot-ի article URL է, թե ոչ։

    Օրինակ՝
    https://www.aravot.am/2026/10/02/1584066/
    """

    if not url:
        return False

    pattern = r"^https://www\.aravot\.am/\d{4}/\d{2}/\d{2}/\d+/?$"

    return bool(re.match(pattern, url))


def normalize_url(url):
    """
    URL-ի վերջում եղած # հատվածը հեռացնում է։
    """

    if not url:
        return None

    url = url.split("#")[0].strip()

    if url.endswith("/"):
        return url

    return url + "/"


def collect_article_urls(category_name, category_url, limit):
    """
    Հավաքում է տվյալ category-ի article URL-երը։
    Էջ առ էջ գնում է մինչև հասնի limit-ին։
    """

    logger.info("=" * 70)
    logger.info(f"Starting category: {category_name}")
    logger.info(f"Target articles: {limit}")

    article_urls = set()

    page_number = 1

    while len(article_urls) < limit:

        if page_number == 1:
            page_url = category_url
        else:
            page_url = urljoin(
                category_url,
                f"page/{page_number}/"
            )

        logger.info(
            f"[{category_name}] Opening category page {page_number}: "
            f"{page_url}"
        )

        try:
            driver.get(page_url)

            time.sleep(1.5)

        except Exception:
            logger.exception(
                f"[{category_name}] Failed to open page {page_number}"
            )
            break

        links = driver.find_elements(By.TAG_NAME, "a")

        logger.info(
            f"[{category_name}] Links found on page "
            f"{page_number}: {len(links)}"
        )

        before_count = len(article_urls)

        for link in links:

            try:
                href = link.get_attribute("href")

                if not href:
                    continue

                href = normalize_url(href)

                if is_article_url(href):
                    article_urls.add(href)

                    if len(article_urls) >= limit:
                        break

            except Exception:
                continue

        new_urls = len(article_urls) - before_count

        logger.info(
            f"[{category_name}] New article URLs: {new_urls}"
        )

        logger.info(
            f"[{category_name}] Total unique articles: "
            f"{len(article_urls)}/{limit}"
        )

        # Եթե հաջորդ էջում այլևս նոր article URL չի գալիս,
        # կանգնում ենք
        if new_urls == 0:
            logger.warning(
                f"[{category_name}] No new article URLs found. "
                f"Stopping pagination."
            )
            break

        page_number += 1

        # Մի քիչ սպասում ենք կայքին
        time.sleep(1.5)

    logger.info(
        f"[{category_name}] URL collection finished: "
        f"{len(article_urls)} articles"
    )

    return list(article_urls)[:limit]


def get_title():
    """
    Փորձում է գտնել article-ի title-ը։
    """

    # Ամենահավանական տարբերակը
    h1_elements = driver.find_elements(By.TAG_NAME, "h1")

    for h1 in h1_elements:

        text = h1.text.strip()

        if text:
            return text

    # Fallback
    title = driver.title.strip()

    return title


def get_article_text():
    """
    Փորձում է գտնել հոդվածի հիմնական տեքստը։

    Մի քանի selector ենք փորձում,
    որովհետև կայքի HTML structure-ը կարող է փոխվել։
    """

    selectors = [
        "article .entry-content",
        "article .post-content",
        "article .article-content",
        "article .content",
        ".entry-content",
        ".post-content",
        ".article-content",
        "article",
        "main"
    ]

    for selector in selectors:

        try:

            elements = driver.find_elements(
                By.CSS_SELECTOR,
                selector
            )

            for element in elements:

                paragraphs = element.find_elements(
                    By.TAG_NAME,
                    "p"
                )

                texts = []

                for p in paragraphs:

                    text = p.text.strip()

                    if text and len(text) > 20:
                        texts.append(text)

                article_text = "\n".join(texts).strip()

                # Եթե բավականաչափ text ունենք
                if len(article_text) >= 200:
                    return article_text

        except Exception:
            continue

    # Վերջնական fallback
    try:

        paragraphs = driver.find_elements(
            By.TAG_NAME,
            "p"
        )

        texts = []

        for p in paragraphs:

            text = p.text.strip()

            if text and len(text) > 20:
                texts.append(text)

        return "\n".join(texts).strip()

    except Exception:

        return ""


def extract_date_from_url(url):
    """
    Article URL-ից ստանում է YYYY-MM-DD։
    """

    match = re.search(
        r"/(\d{4})/(\d{2})/(\d{2})/",
        url
    )

    if match:

        year = match.group(1)
        month = match.group(2)
        day = match.group(3)

        return f"{year}-{month}-{day}"

    return None


def scrape_article(url, category):
    """
    Բացում է մեկ հոդված և վերցնում է տվյալները։
    """

    logger.info(
        f"[{category.upper()}] Opening article: {url}"
    )

    try:

        driver.get(url)

        time.sleep(1.2)

        title = get_title()

        text = get_article_text()

        date = extract_date_from_url(url)

        if not title:
            logger.warning(
                f"[{category.upper()}] Empty title: {url}"
            )

        if not text:
            logger.warning(
                f"[{category.upper()}] Empty article text: {url}"
            )

        if title and text:

            logger.info(
                f"[{category.upper()}] Article collected: "
                f"{title[:80]}"
            )

            return {
                "title": title,
                "text": text,
                "category": category,
                "url": url,
                "date": date
            }

        logger.warning(
            f"[{category.upper()}] Article data incomplete: {url}"
        )

        return None

    except Exception:

        logger.exception(
            f"[{category.upper()}] Failed article: {url}"
        )

        return None


# ============================================================
# MAIN SCRAPER
# ============================================================

all_data = []

try:

    logger.info("=" * 70)
    logger.info("ARAVOT NLP DATASET SCRAPER STARTED")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # 1. COLLECT URLS
    # --------------------------------------------------------

    category_urls = {}

    for category, url in CATEGORIES.items():

        urls = collect_article_urls(
            category,
            url,
            ARTICLES_PER_CATEGORY
        )

        category_urls[category] = urls

        logger.info(
            f"{category.upper()}: {len(urls)} URLs collected"
        )

    # --------------------------------------------------------
    # 2. SCRAPE ARTICLES
    # --------------------------------------------------------

    for category, urls in category_urls.items():

        logger.info("=" * 70)

        logger.info(
            f"START SCRAPING {category.upper()}: "
            f"{len(urls)} articles"
        )

        successful = 0
        failed = 0

        for index, url in enumerate(urls, start=1):

            logger.info(
                f"[{category.upper()}] "
                f"Article {index}/{len(urls)}"
            )

            article = scrape_article(
                url,
                category
            )

            if article:

                all_data.append(article)

                successful += 1

            else:

                failed += 1

            # Rate limit / polite delay
            time.sleep(1)

        logger.info(
            f"[{category.upper()}] Finished."
        )

        logger.info(
            f"[{category.upper()}] "
            f"Successful: {successful}"
        )

        logger.info(
            f"[{category.upper()}] "
            f"Failed: {failed}"
        )

    # --------------------------------------------------------
    # 3. CREATE DATAFRAME
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("Creating DataFrame...")

    df = pd.DataFrame(all_data)

    # Duplicate protection
    if not df.empty:

        df = df.drop_duplicates(
            subset=["url"]
        )

        # Empty rows remove
        df = df.dropna(
            subset=["title", "text", "category"]
        )

        df = df.reset_index(drop=True)

    logger.info(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # 4. DATASET SUMMARY
    # --------------------------------------------------------

    if not df.empty:

        logger.info("Category distribution:")

        counts = df["category"].value_counts()

        for category, count in counts.items():

            logger.info(
                f"  {category}: {count}"
            )

    # --------------------------------------------------------
    # 5. SAVE CSV
    # --------------------------------------------------------

    logger.info(
        f"Saving dataset to: {OUTPUT_FILE}"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        f"Dataset saved successfully: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # 6. FINAL SUMMARY
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info("SCRAPING FINISHED")
    logger.info(f"Total articles: {len(df)}")
    logger.info(f"CSV file: {OUTPUT_FILE}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 70)

except Exception:

    logger.exception(
        "FATAL ERROR IN SCRAPER"
    )

finally:

    driver.quit()

    logger.info(
        "Browser closed."
    )

    logger.info(
        "Program finished."
    )
    import os