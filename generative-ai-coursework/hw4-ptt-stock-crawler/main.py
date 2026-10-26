#!/usr/bin/env python3
"""HW4 - PTT Stock crawler.

CLI:
    python main.py crawl
    python main.py push <start_date> <end_date>
    python main.py popular <start_date> <end_date>
    python main.py keyword <start_date> <end_date> <keyword>

Dates are MMDD (e.g. 0304, 1231). Rename this file to {student_id}.py for
submission.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import httpx
from lxml import html as lhtml


PTT_BASE = "https://www.ptt.cc"
STOCK_INDEX = f"{PTT_BASE}/bbs/Stock/index.html"
COOKIES = {"over18": "1"}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
TZ = datetime.timezone(datetime.timedelta(hours=8))

YEAR_START_TS = int(datetime.datetime(2025, 1, 1, tzinfo=TZ).timestamp())
YEAR_END_TS = int(datetime.datetime(2026, 1, 1, tzinfo=TZ).timestamp())

ARTICLES_FILE = "articles.jsonl"
POPULAR_FILE = "popular_articles.jsonl"

# http(s)://... ending with .jpg/.jpeg/.png/.gif (case-insensitive),
# bounded so the extension is not followed by another alphanumeric char.
IMG_RE = re.compile(
    r'https?://[^\s<>"\'\[\]\(\)（）「」]+?\.(?:jpg|jpeg|png|gif)'
    r'(?![A-Za-z0-9])',
    re.IGNORECASE,
)


def url_timestamp(url: str) -> int | None:
    m = re.search(r"M\.(\d+)\.A", url)
    return int(m.group(1)) if m else None


def date_from_ts(ts: int) -> str:
    dt = datetime.datetime.fromtimestamp(ts, tz=TZ)
    return f"{dt.month:02d}{dt.day:02d}"


def make_client(workers: int = 16) -> httpx.Client:
    return httpx.Client(
        cookies=COOKIES,
        headers=HEADERS,
        timeout=httpx.Timeout(20.0, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=workers * 2,
            max_keepalive_connections=workers,
        ),
    )


def fetch(client: httpx.Client, url: str, retries: int = 3) -> str | None:
    for i in range(retries):
        try:
            r = client.get(url)
            if r.status_code == 200:
                return r.text
            if r.status_code in (404, 403, 410):
                return None
        except httpx.RequestError:
            pass
        time.sleep(0.3 * (i + 1))
    return None


def parse_index_page(html: str) -> tuple[list[dict], str | None]:
    tree = lhtml.fromstring(html)
    out: list[dict] = []
    for div in tree.xpath('//div[contains(@class, "r-ent")]'):
        title_a = div.xpath('./div[@class="title"]/a')
        if not title_a:
            continue
        a = title_a[0]
        title = (a.text_content() or "").strip()
        if not title:
            continue
        href = a.get("href")
        if not href:
            continue
        if "[公告]" in title or title.startswith("Fw:[公告]"):
            continue
        url = urljoin(PTT_BASE, href)
        if not url.startswith("https://"):
            continue
        nrec_list = div.xpath('./div[@class="nrec"]')
        push_text = ""
        if nrec_list:
            push_text = (nrec_list[0].text_content() or "").strip()
        out.append({"title": title, "url": url, "push_text": push_text})

    prev_url: str | None = None
    for btn in tree.xpath('//a[contains(@class, "btn") and contains(@class, "wide")]'):
        if "上頁" in (btn.text_content() or ""):
            href = btn.get("href")
            if href:
                prev_url = urljoin(PTT_BASE, href)
            break
    return out, prev_url


def parse_article(html: str) -> dict | None:
    try:
        tree = lhtml.fromstring(html)
    except Exception:
        return None
    main_list = tree.xpath('//*[@id="main-content"]')
    if not main_list:
        return None
    main = main_list[0]

    full_text = main.text_content()

    pub_idx = full_text.find("※ 發信站")
    body_text = full_text[:pub_idx] if pub_idx >= 0 else None

    comments: list[dict] = []
    for push_div in main.xpath('.//div[contains(concat(" ", normalize-space(@class), " "), " push ")]'):
        tag_list = push_div.xpath('./span[contains(concat(" ", normalize-space(@class), " "), " push-tag ")]')
        uid_list = push_div.xpath('./span[contains(concat(" ", normalize-space(@class), " "), " push-userid ")]')
        if not tag_list or not uid_list:
            continue
        comments.append({
            "tag": (tag_list[0].text_content() or "").strip(),
            "user_id": uid_list[0].text_content() or "",
        })

    image_urls = IMG_RE.findall(full_text)

    return {
        "body_text": body_text,
        "comments": comments,
        "image_urls": image_urls,
    }


def crawl() -> None:
    Path(ARTICLES_FILE).write_text("", encoding="utf-8")
    Path(POPULAR_FILE).write_text("", encoding="utf-8")

    client = make_client()
    articles_f = open(ARTICLES_FILE, "a", encoding="utf-8")
    popular_f = open(POPULAR_FILE, "a", encoding="utf-8")

    try:
        next_url: str | None = STOCK_INDEX
        while next_url:
            html = fetch(client, next_url)
            if html is None:
                print(f"[crawl] failed to fetch {next_url}", file=sys.stderr)
                break
            articles, prev_url = parse_index_page(html)

            page_timestamps: list[int] = []
            for art in articles:
                ts = url_timestamp(art["url"])
                if ts is None:
                    continue
                page_timestamps.append(ts)
                if ts >= YEAR_END_TS or ts < YEAR_START_TS:
                    continue
                entry = {
                    "date": date_from_ts(ts),
                    "title": art["title"],
                    "url": art["url"],
                }
                articles_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if art["push_text"] == "爆":
                    popular_f.write(
                        json.dumps(entry, ensure_ascii=False) + "\n"
                    )

            articles_f.flush()
            popular_f.flush()

            if page_timestamps and all(t < YEAR_START_TS for t in page_timestamps):
                break
            next_url = prev_url
    finally:
        articles_f.close()
        popular_f.close()
        client.close()


def load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def fetch_articles_parallel(
    urls: list[str], workers: int = 24,
) -> list[tuple[str, dict | None]]:
    if not urls:
        return []
    client = make_client(workers=workers)
    results: list[tuple[str, dict | None]] = []

    def task(url: str) -> tuple[str, dict | None]:
        html = fetch(client, url)
        if html is None:
            return url, None
        try:
            return url, parse_article(html)
        except Exception:
            return url, None

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(task, u) for u in urls]
            for fut in as_completed(futures):
                results.append(fut.result())
    finally:
        client.close()
    return results


def push_cmd(start_date: str, end_date: str) -> None:
    articles = load_jsonl(ARTICLES_FILE)
    in_range = [a for a in articles if start_date <= a["date"] <= end_date]
    urls = [a["url"] for a in in_range]

    push_total = 0
    boo_total = 0
    push_counter: Counter[str] = Counter()
    boo_counter: Counter[str] = Counter()

    for _, parsed in fetch_articles_parallel(urls):
        if parsed is None:
            continue
        for c in parsed["comments"]:
            if c["tag"] == "推":
                push_total += 1
                push_counter[c["user_id"]] += 1
            elif c["tag"] == "噓":
                boo_total += 1
                boo_counter[c["user_id"]] += 1

    def top10(counter: Counter[str]) -> list[dict]:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        return [{"user_id": uid, "count": cnt} for uid, cnt in items]

    out = {
        "push": {"total": push_total, "top10": top10(push_counter)},
        "boo": {"total": boo_total, "top10": top10(boo_counter)},
    }

    fname = f"push_{start_date}_{end_date}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def popular_cmd(start_date: str, end_date: str) -> None:
    articles = load_jsonl(POPULAR_FILE)
    in_range = [a for a in articles if start_date <= a["date"] <= end_date]
    urls = [a["url"] for a in in_range]

    image_urls: list[str] = []
    for _, parsed in fetch_articles_parallel(urls):
        if parsed is None:
            continue
        image_urls.extend(parsed["image_urls"])

    out = {
        "number_of_popular_articles": len(in_range),
        "image_urls": image_urls,
    }

    fname = f"popular_{start_date}_{end_date}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def keyword_cmd(start_date: str, end_date: str, keyword: str) -> None:
    articles = load_jsonl(ARTICLES_FILE)
    in_range = [a for a in articles if start_date <= a["date"] <= end_date]
    urls = [a["url"] for a in in_range]

    image_urls: list[str] = []
    for _, parsed in fetch_articles_parallel(urls):
        if parsed is None:
            continue
        if parsed["body_text"] is None:
            continue
        if keyword not in parsed["body_text"]:
            continue
        image_urls.extend(parsed["image_urls"])

    out = {"image_urls": image_urls}

    fname = f"keyword_{start_date}_{end_date}_{keyword}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("crawl")

    p = sub.add_parser("push")
    p.add_argument("start_date")
    p.add_argument("end_date")

    p = sub.add_parser("popular")
    p.add_argument("start_date")
    p.add_argument("end_date")

    p = sub.add_parser("keyword")
    p.add_argument("start_date")
    p.add_argument("end_date")
    p.add_argument("keyword")

    args = parser.parse_args()

    if args.cmd == "crawl":
        crawl()
    elif args.cmd == "push":
        push_cmd(args.start_date, args.end_date)
    elif args.cmd == "popular":
        popular_cmd(args.start_date, args.end_date)
    elif args.cmd == "keyword":
        keyword_cmd(args.start_date, args.end_date, args.keyword)


if __name__ == "__main__":
    main()
