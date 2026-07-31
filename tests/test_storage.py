from src.storage import SnapshotStore


def test_new_secret_key_is_only_sent_as_apikey() -> None:
    store = SnapshotStore("https://example.supabase.co", "sb_secret_example")

    assert store.headers["apikey"] == "sb_secret_example"
    assert "Authorization" not in store.headers


def test_legacy_service_role_key_remains_supported() -> None:
    store = SnapshotStore("https://example.supabase.co", "legacy.jwt.key")

    assert store.headers["apikey"] == "legacy.jwt.key"
    assert store.headers["Authorization"] == "Bearer legacy.jwt.key"
