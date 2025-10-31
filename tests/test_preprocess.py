from src.utils import preprocess_text


def test_preprocess_basic():
    s = "Hello WORLD!! Visit http://example.com @user"
    out = preprocess_text(s)
    # ensure lowercased and url/handles removed
    assert 'http' not in out
    assert '@user' not in out
    assert out == out.lower()
