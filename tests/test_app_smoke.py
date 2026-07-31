from streamlit.testing.v1 import AppTest


def test_app_renders_without_credentials() -> None:
    app = AppTest.from_file("app.py", default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Options Positioning Dashboard"
    assert app.info
