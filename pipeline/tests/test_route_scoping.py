from pipeline.stages.cleaners.route_scoping import (
    document_title_from_html,
    scope_route_specific_html,
)


def test_scopes_faq_detail_route_to_its_active_panel():
    raw = """
    <html><head><title>Student clubs | MBZUAI</title>
    <meta property="og:site_name" content="MBZUAI">
    <meta property="og:title" content="Student clubs">
    </head><body><main>
      <div data-pc-name="accordionpanel" data-p-active="false" id="housing">
        <button>Does MBZUAI provide housing?</button><div><p>Housing answer.</p></div>
      </div>
      <div data-pc-name="accordionpanel" data-p-active="true" id="student-clubs">
        <button>What clubs are available?</button>
        <div><p>Students can join existing clubs or establish a new club.</p></div>
      </div>
    </main></body></html>
    """

    result = scope_route_specific_html(
        raw,
        ["https://preprod.mbzuai.ac.ae/faq/student-clubs"],
    )

    assert result.method == "faq_active_panel"
    assert result.document_title == "What clubs are available?"
    assert "establish a new club" in result.html
    assert "Housing answer" not in result.html


def test_scopes_pre_rendered_spa_panels_by_route_slug():
    raw = """
    <html><head>
      <meta property="og:title" content="Sustainability Technologies From Lab to Market">
    </head><body>
      <div class="panel-content" id="another-event">
        <h3>Another event</h3><p>This is unrelated event content with enough words.</p>
      </div>
      <div class="panel-content" id="sustainability-technologies-from-lab-to-market">
        <h3>Sustainability Technologies From Lab to Market</h3>
        <p>Professor Yi Cui discusses batteries and sustainability materials technologies.</p>
      </div>
    </body></html>
    """

    result = scope_route_specific_html(
        raw,
        [
            "https://ai-nexus.mbzuai.ac.ae/distinguished-lecture-series/"
            "sustainability-technologies-from-lab-to-market"
        ],
    )

    assert result.method == "route_id_panel"
    assert "Professor Yi Cui" in result.html
    assert "Another event" not in result.html


def test_scopes_repeated_records_by_exact_page_title_when_no_route_id_exists():
    raw = """
    <html><head><meta property="og:title" content="Target lecture"></head><body>
      <article data-title="Other lecture"><h3>Other</h3><p>Other lecture details live here.</p></article>
      <article data-title="Target lecture"><h3>Target lecture</h3>
        <p>The target speaker and abstract are present in this record.</p></article>
    </body></html>
    """

    result = scope_route_specific_html(
        raw,
        ["https://events.example.edu/lectures/target-lecture"],
    )

    assert result.method == "title_matched_record"
    assert "target speaker" in result.html
    assert "Other lecture details" not in result.html


def test_conventional_page_is_not_destructively_scoped():
    raw = """
    <html><head><title>Research overview</title></head><body><main>
      <h1>Research overview</h1><p>The complete conventional page remains intact.</p>
    </main></body></html>
    """

    result = scope_route_specific_html(
        raw,
        ["https://www.example.edu/research/overview"],
    )

    assert result.method == ""
    assert result.html == raw


def test_document_title_removes_declared_site_name_suffix():
    raw = """
    <html><head><title>Admissions | Example University</title>
      <meta property="og:site_name" content="Example University">
    </head></html>
    """

    assert document_title_from_html(raw) == "Admissions"
