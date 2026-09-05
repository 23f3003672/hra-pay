"""Smoke test for the Streamlit dashboard.

The dashboard is what gets demonstrated, so a silent break in it is a break in
the deliverable. AppTest actually executes the script — every tab, every widget
callback — and surfaces any exception that would otherwise only appear as a red
box on screen during the demo.

Skipped rather than failed when streamlit is not installed, since it is an
optional extra and the core library must remain testable without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "dashboard.py"

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="dashboard extra not installed"
)


@pytest.fixture(scope="module")
def app():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=300)
    return at.run()


def test_dashboard_renders_without_exceptions(app) -> None:
    """The one that matters. Any traceback here is a red box during the demo."""
    assert not app.exception, [str(e.value) for e in app.exception]


def test_dashboard_has_all_four_views(app) -> None:
    assert len(app.tabs) == 4


def test_dashboard_shows_the_episode_controls(app) -> None:
    labels = [s.label for s in app.selectbox]
    assert "Policy" in labels
    assert "Distribution" in labels


def test_dashboard_renders_content_not_just_placeholders(app) -> None:
    """Guards against the tabs loading but every panel bailing to an st.info()."""
    assert len(app.metric) >= 6
    assert len(app.dataframe) >= 1
