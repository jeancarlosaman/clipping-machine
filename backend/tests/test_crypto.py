"""Pure unit tests for app.core.crypto -- no DB/IO needed."""
from app.core.crypto import decrypt_token, encrypt_token


def test_encrypt_then_decrypt_round_trips():
    plaintext = "super-secret-tiktok-access-token"
    ciphertext = encrypt_token(plaintext)
    assert ciphertext != plaintext
    assert decrypt_token(ciphertext) == plaintext


def test_ciphertext_is_not_plaintext_substring():
    # A weak/no-op "encryption" might still leak the token as a substring
    # of the stored value -- guard against that specifically.
    plaintext = "super-secret-tiktok-access-token"
    ciphertext = encrypt_token(plaintext)
    assert plaintext not in ciphertext


def test_dev_default_key_is_derived_from_jwt_secret(monkeypatch):
    # With CREATOR_ACCOUNT_ENCRYPTION_KEY unset (the default), a different
    # jwt_secret must produce a key that can't decrypt tokens encrypted
    # under a different jwt_secret -- proves the derivation actually depends
    # on the secret rather than being a fixed/ignored value.
    monkeypatch.setattr("app.core.config.settings.creator_account_encryption_key", "")
    monkeypatch.setattr("app.core.config.settings.jwt_secret", "secret-one")
    ciphertext = encrypt_token("a-token")

    monkeypatch.setattr("app.core.config.settings.jwt_secret", "secret-two")
    try:
        decrypt_token(ciphertext)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_explicit_key_takes_precedence_over_derived_one():
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode("utf-8")
    import app.core.crypto as crypto_module

    original = crypto_module.settings.creator_account_encryption_key
    crypto_module.settings.creator_account_encryption_key = key
    try:
        ciphertext = encrypt_token("a-token")
        assert decrypt_token(ciphertext) == "a-token"
    finally:
        crypto_module.settings.creator_account_encryption_key = original
