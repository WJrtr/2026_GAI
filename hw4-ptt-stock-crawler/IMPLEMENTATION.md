# HW4 Implementation Notes

How `main.py` solves the four PTT Stock crawler subcommands, with the design choices and tradeoffs that drove each decision.

---

## Stack

- **Python 3.12** stdlib (`argparse`, `datetime`, `json`, `re`, `concurrent.futures`, `urllib.parse`, `collections.Counter`, `pathlib`)
- **`httpx`** — HTTP client with connection pooling and keepalive
- **`lxml.html`** — direct parsing (no BeautifulSoup wrapper)

`bs4` is in `requirements.txt` per the PDF but unused — `lxml` directly is ~5× faster than `BeautifulSoup(html, "lxml")`, which mattered for the 10-min per-test budget.

---

## 1. `crawl` — walk the index pages backward

```
python main.py crawl
```

### Algorithm

1. Start at `https://www.ptt.cc/bbs/Stock/index.html` (latest page).
2. Parse each `div.r-ent`: title, href, and `nrec` text (push count or `爆`).
3. For each article, **derive its date and year from the URL timestamp** — the `M.<ts>.A.<id>.html` segment encodes Unix epoch.
4. Keep articles where `YEAR_START_TS ≤ ts < YEAR_END_TS` (Taiwan timezone, UTC+8).
5. Append matching entries to `articles.jsonl`; if `nrec == "爆"`, also append to `popular_articles.jsonl`.
6. Follow the `‹ 上頁` link until a page has **all** timestamps `< YEAR_START_TS` — then stop.

### Why URL timestamp instead of the listing-page `MM/DD`?

The listing only shows month/day, ambiguous between 2024/2025 and 2025/2026 for crossing-year articles. The URL timestamp is unambiguous Unix epoch and survives DST/timezone confusion.

```python
YEAR_START_TS = int(datetime(2025, 1, 1, tzinfo=TZ).timestamp())  # 1735660800
YEAR_END_TS   = int(datetime(2026, 1, 1, tzinfo=TZ).timestamp())  # 1767196800
```

### Filtering rules (per spec)

- Skip titles containing `[公告]` or starting with `Fw:[公告]`.
- Skip rows with no anchor (deleted articles).
- Skip empty titles.
- URLs must be `https://`.
- Use **listing-page title**, not in-article title.

### File handling

`articles.jsonl` and `popular_articles.jsonl` are **truncated then opened in append mode**, with `.flush()` after each page. The PDF's note 4 explicitly recommends this so partial progress survives an exception mid-crawl.

### Performance

Sequential walk: **28 seconds** for the full year, ~1230 index pages, 28564 articles. HTTP/1.1 keepalive on a single `httpx.Client` does the heavy lifting — no parallelism needed for `crawl`.

---

## 2. `push` — count comments + top-10 user IDs

```
python main.py push <start> <end>
```

### Algorithm

1. Load `articles.jsonl`, filter by `start ≤ date ≤ end` (string compare on MMDD works because zero-padded).
2. Concurrently fetch each article via `ThreadPoolExecutor(max_workers=24)` sharing one `httpx.Client`.
3. Parse `div.push` blocks. Each has `span.push-tag` (推 / 噓 / →) and `span.push-userid`.
4. Tally totals and per-user counts in two `collections.Counter`s.
5. Top 10: `sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:10]` — **count desc, then user_id asc** for deterministic tie-breaking.

### The user_id padding gotcha

PTT pads user_ids to **12 characters** with trailing spaces (legacy BBS fixed-width display). The answer keeps these spaces. My first version called `get_text(strip=True)` and stripped them, scoring **0.0 IoU** despite correct counts. Fix: `uid_list[0].text_content() or ""` — no strip.

### XPath multi-class trap

PTT push spans have multiple classes: `<span class="hl push-tag">`, `<span class="f3 hl push-userid">`. My initial XPath `./span[@class="push-tag"]` requires **exact** class match and matched zero elements. Replaced with the canonical class-token check:

```xpath
./span[contains(concat(" ", normalize-space(@class), " "), " push-tag ")]
```

This matches when `push-tag` is a complete class token, regardless of position.

---

## 3. `popular` — count `推爆` articles + collect images

```
python main.py popular <start> <end>
```

Read `popular_articles.jsonl` (already filtered to `推爆` during crawl), filter by date range, fetch each, run image regex on the full article text. Output:

```json
{ "number_of_popular_articles": N, "image_urls": [...] }
```

Duplicates aren't deduped (per spec).

---

## 4. `keyword` — articles whose body contains a substring

```
python main.py keyword <start> <end> <keyword>
```

### Body-scope rule

Per spec, the body is **from `作者` (inclusive) to `※ 發信站` (exclusive)**. If `※ 發信站` is missing, the article is skipped.

```python
full_text = main.text_content()
pub_idx = full_text.find("※ 發信站")
body_text = full_text[:pub_idx] if pub_idx >= 0 else None
```

`text_content()` of the entire `#main-content` starts with the author metaline (作者), so slicing up to `pub_idx` gives the correct body scope.

### Image extraction

Images are collected from the **whole article (body + comments)**, not just the body. Body is only used for the `keyword in body_text` membership check.

### Image URL regex

```python
IMG_RE = re.compile(
    r'https?://[^\s<>"\'\[\]\(\)（）「」]+?\.(?:jpg|jpeg|png|gif)(?![A-Za-z0-9])',
    re.IGNORECASE,
)
```

- Non-greedy match so `a.jpg.gif` cleaves at the first valid extension.
- Negative lookahead `(?![A-Za-z0-9])` ensures the extension is at a token boundary so `.jpg` doesn't match `.jpgfoo`.
- Stops at quotes, brackets, and Chinese punctuation that often abuts URLs in comments.

---

## Performance journey

| Iteration | Push 0304-0430 (7238 articles) |
|---|---:|
| `BeautifulSoup(html, "lxml")` + 24 workers | **4 min 52 s** |
| `lxml.html.fromstring` direct + 24 workers | **49 s** (initial test, parsing broken — returned 0) |
| Same after XPath multi-class fix | **79 s** (correct) |

The BS-to-lxml swap was driven by a profiling experiment:

```
sequential parse (BS):     5.8 s / 416 articles  (14 ms each)
8-thread parse (BS):      12.9 s / 416 articles  (slower! GIL contention)
sequential parse (lxml):   1.2 s / 416 articles  ( 3 ms each)
```

`lxml`'s C-level `text_content()` and XPath release the GIL more cleanly than BeautifulSoup's traversals. Per-article time dropped from ~14 ms to ~3 ms.

---

## Validation against `2025_answer/`

```
python eval.py 2025_answer .
```

| Test | IoU | Threshold | Notes |
|---|---:|---:|---|
| crawl | 1.0000 | 0.95 | 28564/28564 articles match exactly |
| push 0304 0430 | 1.0000 | 0.80 | totals match: push=418924, boo=65649 |
| popular 0401 0731 | 1.0000 | 0.95 | 17 popular articles, all image URLs match |
| keyword 0504 0831 台股 | 0.9941 | 0.95 | minor drift (PTT edits/deletes since answer was generated) |

All within time budgets:

| Command | Wall time | Limit |
|---|---:|---:|
| crawl | 28 s | 20 min |
| push (58 days) | 79 s | 10 min |
| popular (122 days) | 0.6 s | 10 min |
| keyword (120 days) | 100 s | 10 min |

---

## Things that didn't work / were considered

- **BeautifulSoup**: too slow under threaded load. Replaced with `lxml` direct.
- **Listing-page date string**: ambiguous at year boundaries. Replaced with URL timestamp.
- **`get_text(strip=True)` for user_id**: stripped the 12-char padding the answer expects. Replaced with raw `text_content()`.
- **`@class="push-tag"` XPath**: exact-match doesn't work for multi-class spans. Replaced with class-token containment.
- **Multiprocessing for parsing**: process startup cost dwarfs the savings once `lxml` reduced parse time to 3 ms/article.
- **Binary search for index page range**: once sequential walk hit 28 s with keepalive, optimization unnecessary.

---

## Submission

Rename `main.py` to `{student_id}.py`. The script reads/writes only the four files specified by the PDF in the current working directory and depends solely on stdlib + `httpx` + `lxml`, both pinned in `requirements.txt`.
