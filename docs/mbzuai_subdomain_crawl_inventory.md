# MBZUAI public-origin crawl inventory

Reviewed: 2026-08-20 (Asia/Dubai)
Configuration revision: `2026-08-20-v3`

This inventory separates public institutional content from every hostname that
merely appears in DNS, certificate transparency, or the main site's link graph.
Only explicitly approved hosts are eligible for production egress.

## Included in the normal production crawl

| Origin | Discovery | Observed inventory | Run policy |
| --- | --- | ---: | --- |
| `mbzuai.ac.ae` | XML sitemap and robots declaration | 2,664 raw; 2,205 eligible after fresh empty-cohort verification | Sitemap seeds |
| `careers.mbzuai.ac.ae` | WordPress `wp-sitemap.xml` | 49 raw; 35 HTTP 200 and 14 stale HTTP 404 on review | Sitemap seeds; only successful pages count toward crawl coverage |
| `ifm.ai` | Yoast sitemap index | 22 URLs | Sitemap seeds; canonical target of `ifm.mbzuai.ac.ae` |
| `research.mbzuai.ac.ae` | Robots-declared XML sitemap | 10 URLs | Sitemap seeds |
| `ai-nexus.mbzuai.ac.ae` | Robots-declared Yoast sitemap | 10 URLs | Sitemap seeds |
| `hpp.mbzuai.ac.ae` | Yoast sitemap index | 4 URLs | Sitemap seeds |
| `library.mbzuai.ac.ae` | No usable sitemap | 19 live pages in the validation crawl | Root seed plus same-host bounded link discovery; minimum 15 successful pages |
| `metaverse.mbzuai.ac.ae` | No usable sitemap | 38 live pages in the validation crawl | Root seed plus same-host bounded link discovery; minimum 30 successful pages |
| `buildit.mbzuai.ac.ae` | No usable sitemap; robots path returns a real HTTP 404 | 7 live pages in the validation crawl | Retain the substantive root only; minimum 1 successful page |

`www.mbzuai.ac.ae` and `ifm.mbzuai.ac.ae` are approved redirect aliases, but
they are not independent required content inventories.

## Deferred specialized crawl

| Origin | Reason |
| --- | --- |
| `irep.mbzuai.ac.ae` | Public DSpace repository with roughly 2,714 sitemap URLs. Its robots policy requests `Crawl-delay: 10`, so it must use a dedicated low-rate ingester rather than the normal concurrent browser batch. The normal crawl still captures the Library's public repository landing page. |

## Explicit policy or safety exclusions

| Origin/class | Reason |
| --- | --- |
| `academy.mbzuai.ac.ae` | `robots.txt` disallows `/` for every user agent. |
| `dclibrary.mbzuai.ac.ae` | Presented TLS certificate does not match the hostname. |
| `xing.mbzuai.ac.ae` | `robots.txt` returns HTTP 403, so crawl permission cannot be established. |
| `apply.mbzuai.ac.ae`, `engage.mbzuai.ac.ae` | Applicant/authenticated workflows, not public institutional content pages. |
| `staticcdn.mbzuai.ac.ae` and marketing click/view/image hosts | Assets or tracking endpoints, not a page corpus. |
| `*.hci.mbzuai.ac.ae` service hosts | Administrative, notebook, database, proxy, storage, observability, and other research-service endpoints discovered passively; never eligible by domain suffix. |
| Stale/unverified hosts such as `blog`, `bayesian`, and `beltrame` | Failed, parked, stale, or not linked as approved institutional content during review. |

## Enforced run contract

The production crawler now enforces:

1. an exact `allowed_hosts` set in addition to legacy domain suffixes;
2. multi-origin robots and sitemap discovery;
3. per-host minimum sitemap URL counts before browser startup;
4. same-host-only, depth- and page-bounded discovery for approved sites without sitemaps; and
5. live per-origin robots policies on the browser frontier, raw HTTP fallback,
   every redirect hop, and same-site downloads; and
6. per-host successful page minimums before the crawl stage can complete.

The smaller `mbzuai_subdomains_artifacts` profile exercises the same subdomain
contract without re-crawling the main MBZUAI sitemap and contains only the
artifact collection stage.

## Artifact collection validation run

Run `mbzuai-subdomains-artifacts-20260820-v9` completed the crawler stage on
2026-08-20 and passed an independent run audit with zero errors and zero
warnings. It saved 145 successful page captures, 131 accepted crawler Markdown
sidecars, and one PDF:

| Host | Successful pages |
| --- | ---: |
| `careers.mbzuai.ac.ae` | 35 |
| `research.mbzuai.ac.ae` | 10 |
| `ai-nexus.mbzuai.ac.ae` | 10 |
| `hpp.mbzuai.ac.ae` | 4 |
| `ifm.ai` | 22 |
| `library.mbzuai.ac.ae` | 19 |
| `metaverse.mbzuai.ac.ae` | 38 |
| `buildit.mbzuai.ac.ae` | 7 |

The run discovered 95 sitemap URLs. It classified 14 stale Careers entries as
HTTP 404, 42 broken Metaverse publication links as HTTP 404, and three malformed
Metaverse links as HTTP 400; none were saved as successful pages. Five robots
policy encounters were blocked, and no Library login/search URL appears in
successful page metadata.

BuildIt's browser-rendered pages currently raise a client-side exception. The
crawler recognizes that error shell and retains the substantive root from its
healthy raw source. Cleaning review confirmed that the six child routes
(`about`, `apply`, `highlights`, `benefits`, `network`, and `faqs`) contain no
server-rendered visible content and are not required for the chatbot. They are
therefore no longer seeded, added to the crawl frontier, or counted as future
coverage requirements.

Only the crawler stage ran. No cleaner, converter, chunker, embedder, or upload
stage was executed.
