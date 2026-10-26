# HW4 — PTT Stock Board Crawler

`web crawling` `httpx` `lxml / XPath` `concurrency` `profiling & optimization` `CLI tooling`

## Assignment

Write a single Python script that crawls **every 2025 article on the PTT Stock board** and exposes four CLI subcommands: `crawl` (all articles + 推爆 articles to JSONL), `push` (push/boo counts and top-10 commenters in a date range), `popular` (推爆 articles and their image URLs), and `keyword` (articles whose body contains a keyword, plus image URLs). Graded by IoU against reference outputs under strict time limits.

> Summarized from the course assignment spec (NYCU Generative AI, Spring 2026). The original spec PDFs are not redistributed.


A single-file CLI that crawls every 2025 post on the PTT Stock board and answers four query types. See **[IMPLEMENTATION.md](IMPLEMENTATION.md)** for design decisions, pitfalls found, and the profiling that drove them.

## Commands

```bash
pip install -r requirements.txt        # only httpx and lxml are actually used
python main.py crawl                                   # → articles.jsonl, popular_articles.jsonl
python main.py push    0304 0430                       # push/boo totals + top-10 users
python main.py popular 0401 0731                       # 推爆 articles + image URLs
python main.py keyword 0504 0831 台股                   # articles whose body contains keyword + image URLs
```

## Results (vs. reference answers)

| Task | IoU | Wall time | Limit |
|---|---:|---:|---:|
| crawl (28,564 articles) | 1.0000 | 28 s | 20 min |
| push 0304–0430 | 1.0000 | 79 s | 10 min |
| popular 0401–0731 | 1.0000 | 0.6 s | 10 min |
| keyword 0504–0831 台股 | 0.9941 | 100 s | 10 min |

Highlights: dates derived from the URL's Unix timestamp (no year ambiguity), `httpx` keep-alive for a 28 s sequential crawl, and switching BeautifulSoup → raw `lxml` cut per-article parse time from ~14 ms to ~3 ms (push task 4m52s → 79s).

`results/` contains the generated outputs (the full `articles.jsonl` is omitted for size; regenerate with `crawl`).
