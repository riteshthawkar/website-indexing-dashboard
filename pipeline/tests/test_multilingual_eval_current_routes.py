from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = REPO_ROOT / "scripts" / "build_multilingual_eval_v2.py"
    spec = importlib.util.spec_from_file_location("build_multilingual_eval_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preprod_host_is_not_misclassified_as_a_subdomain():
    module = _module()

    assert module._is_main_host("mbzuai.ac.ae")
    assert module._is_main_host("preprod.mbzuai.ac.ae")
    assert not module._is_main_host("careers.mbzuai.ac.ae")


def test_identical_route_aliases_prefer_current_public_route():
    module = _module()
    pages = [
        {
            "source_url": "https://preprod.mbzuai.ac.ae/study/phd-programs",
            "full_text": "The same rendered program content.",
            "language": "English",
        },
        {
            "source_url": "https://preprod.mbzuai.ac.ae/academics/phd-programs",
            "full_text": "The same rendered program content.",
            "language": "English",
        },
        {
            "source_url": "https://preprod.mbzuai.ac.ae/ar/node/335",
            "full_text": "The same rendered program content.",
            "language": "Arabic",
        },
    ]

    selected = module._deduplicate_identical_pages(pages)

    assert [row["source_url"] for row in selected] == [
        "https://preprod.mbzuai.ac.ae/academics/phd-programs",
        "https://preprod.mbzuai.ac.ae/ar/node/335",
    ]


def test_current_route_lookup_falls_back_only_when_needed():
    module = _module()
    pages = [{"source_url": "https://preprod.mbzuai.ac.ae/about-us"}]

    selected = module._find_page_first(
        pages,
        "https://preprod.mbzuai.ac.ae/about-us/mission",
        "https://preprod.mbzuai.ac.ae/about-us",
    )

    assert selected is pages[0]


def test_current_pdf_lookup_uses_first_available_title():
    module = _module()
    pdfs = [{"title": "MBZUAI Campus Map V1044331768"}]

    selected = module._find_pdf_first_title(
        pdfs,
        "Removed Academic Catalogue",
        "MBZUAI Campus Map V1044331768",
    )

    assert selected is pdfs[0]


def test_subdomain_floor_replaces_only_an_eligible_main_host_fact():
    module = _module()
    tasks = [
        module._task(
            "en-fact-001",
            language="English",
            query_type="fact",
            source_type="webpage",
            packs=[
                {
                    "source_key": "main",
                    "source_url": "https://preprod.mbzuai.ac.ae/about-us",
                    "host": "preprod.mbzuai.ac.ae",
                }
            ],
        ),
        module._task(
            "en-scoped-001",
            language="English",
            query_type="scoped",
            source_type="webpage",
            packs=[
                {
                    "source_key": "main-scoped",
                    "source_url": "https://preprod.mbzuai.ac.ae/admissions",
                    "host": "preprod.mbzuai.ac.ae",
                }
            ],
        ),
    ]
    candidate = {
        "source_key": "library",
        "source_url": "https://library.mbzuai.ac.ae/opening-hours",
        "host": "library.mbzuai.ac.ae",
        "page_type": "content",
        "word_count": 1000,
    }

    selected = module._ensure_subdomain_task_floor(
        tasks,
        candidates=[candidate],
        minimum=1,
    )

    assert selected[0]["sources"] == [candidate]
    assert selected[1] is tasks[1]


def test_navigation_queries_must_name_their_owning_page():
    module = _module()
    sources = [{"title": "Research Scientist – Gender & AI – Careers"}]

    assert not module._navigation_query_has_source_identity(
        "من صفحة الوظيفة هذه، إلى أين يقود إجراء التقديم؟",
        sources,
    )
    assert module._navigation_query_has_source_identity(
        "في صفحة Research Scientist – Gender & AI، إلى أين يقود إجراء التقديم؟",
        sources,
    )


def test_navigation_task_accepts_visually_indistinguishable_actions():
    module = _module()
    shared = {
        "label": "[email protected]",
        "context_label": "Contact us",
        "source_section_heading": "Contact us",
        "action_type": "contact",
    }
    source = {
        "source_key": "pathway",
        "actions": [
            {**shared, "action_id": "action:visitor", "target_url": "https://example.test/a"},
            {**shared, "action_id": "action:programs", "target_url": "https://example.test/b"},
        ],
    }

    task = module._task(
        "ar-navigation-001",
        language="Arabic",
        query_type="scoped",
        source_type="webpage",
        packs=[source],
        navigation=True,
    )

    assert task["required_action_ids"] == ["action:programs", "action:visitor"]
