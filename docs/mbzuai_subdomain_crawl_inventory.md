# MBZUAI public-origin crawl inventory

Reviewed: 2026-08-20 (Asia/Dubai)
Configuration revision: `2026-08-20-v1`

This inventory separates public institutional content from every hostname that
merely appears in DNS, certificate transparency, or the main site's link graph.
Only explicitly approved hosts are eligible for production egress.

## Included in the normal production crawl

| Origin | Discovery | Observed inventory | Run policy |
| --- | --- | ---: | --- |
| `mbzuai.ac.ae` | XML sitemap and robots declaration | 2,664 raw; 2,205 eligible after fresh empty-cohort verification | Sitemap seeds |
| `careers.mbzuai.ac.ae` | WordPress `wp-sitemap.xml` | 49 URLs | Sitemap seeds |
| `ifm.ai` | Yoast sitemap index | 22 URLs | Sitemap seeds; canonical target of `ifm.mbzuai.ac.ae` |
| `research.mbzuai.ac.ae` | Robots-declared XML sitemap | 10 URLs | Sitemap seeds |
| `ai-nexus.mbzuai.ac.ae` | Robots-declared Yoast sitemap | 10 URLs | Sitemap seeds |
| `hpp.mbzuai.ac.ae` | Yoast sitemap index | 4 URLs | Sitemap seeds |
| `library.mbzuai.ac.ae` | No usable sitemap | Not fixed | Root seed plus same-host bounded link discovery |
| `metaverse.mbzuai.ac.ae` | No usable sitemap | Not fixed | Root seed plus same-host bounded link discovery |
| `buildit.mbzuai.ac.ae` | No usable sitemap; robots path returns a real HTTP 404 | Not fixed | Root seed plus same-host bounded link discovery |

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
