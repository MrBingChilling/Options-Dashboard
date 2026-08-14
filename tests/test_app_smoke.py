from streamlit.testing.v1 import AppTest


def test_app_renders_without_credentials() -> None:
    app = AppTest.from_file("app.py", default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "IV & Skew"
    assert app.info


def test_ai_summary_view_renders_without_credentials() -> None:
    app = AppTest.from_file("app.py", default_timeout=20).run()
    assert app.segmented_control[0].options == ["IV & Skew", "Gamma & Volume", "AI Summary"]
    app.segmented_control[0].set_value("AI Summary").run()
    assert not app.exception
    assert app.title[0].value == "IV & Skew"
    assert app.subheader[0].value == "Daily AI Summary"
    assert app.info
