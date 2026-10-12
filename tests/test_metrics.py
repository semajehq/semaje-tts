from semaje_tts.evals.metrics import cer


def test_cer():
    assert cer("habari", "habari") == 0
    assert cer("abcd", "abed") == 0.25
    assert cer("Habari, yako!", "habari yako") == 0
    assert cer("", "x") == 1.0
