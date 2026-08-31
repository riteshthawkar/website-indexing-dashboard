from pipeline.core.admissions_routing import (
    admissions_surface_preference,
    admissions_workflow_audience,
    canonical_admissions_marker,
)


def test_admissions_workflow_classifier_covers_degree_levels_and_arabic():
    assert admissions_workflow_audience("How do I apply for a PhD at MBZUAI?") == "phd"
    assert admissions_workflow_audience("What is the MSc application process?") == "masters"
    assert admissions_workflow_audience("كيفية التقديم لبرنامج الدكتوراه؟") == "phd"
    assert canonical_admissions_marker("How can I start an undergraduate application?") == (
        "/admissions/undergraduate-admissions"
    )


def test_non_admissions_apply_language_does_not_force_admissions_pages():
    assert admissions_workflow_audience(
        "Who can apply for onsite access to the MBZUAI Library?"
    ) == ""
    assert canonical_admissions_marker("How do I apply for an MBZUAI job vacancy?") == ""


def test_evergreen_workflow_penalizes_news_but_time_bound_question_does_not():
    news = {
        "source_url": (
            "https://preprod.mbzuai.ac.ae/knowledge-center/the-node/"
            "mbzuai-opens-admissions-fall-2026-intake"
        ),
        "title": "MBZUAI opens admissions for Fall 2026 intake",
        "page_type": "news_or_event",
    }
    assert admissions_surface_preference(
        "How do I apply for a PhD at MBZUAI?", **news
    ) == -1.0
    assert admissions_surface_preference(
        "How do I apply for the Fall 2026 PhD intake?", **news
    ) == -0.10
