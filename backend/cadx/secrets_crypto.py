"""At-rest encryption for DB-persisted secrets.

Runtime secrets (LLM/Tripo/image API keys, the Hugging Face token) live in the
``app_settings`` JSON column. Values are Fernet-encrypted (AES-128-CBC + HMAC)
before they hit the database; the key itself never enters the DB. Resolution
order:

1. ``CADX_SETTINGS_KEY`` env var — a Fernet key (urlsafe base64, 32 bytes).
   Generate with ``python -c "from cryptography.fernet import Fernet;
   print(Fernet.generate_key().decode())"``.
2. A key file at ``{data_dir}/.settings.key`` — created once, mode 0600.
   ``data_dir`` is a shared/persistent volume in the docker stack, so api and
   worker containers transparently share it.

Stored values carry the ``enc:v1:`` prefix; anything without it is legacy
plaintext (pre-encryption deployments) and is transparently upgraded to the
encrypted form on the next settings save.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import structlog
from cryptography.fernet import Fernet, InvalidToken

log = structlog.get_logger()

SECRET_PREFIX = "enc:v1:"
_KEY_FILE = ".settings.key"

_fernet: Fernet | None = None


def _resolve_key() -> bytes:
    env = os.environ.get("CADX_SETTINGS_KEY", "").strip()
    if env:
        return env.encode()
    from cadx.config import get_settings

    path = Path(get_settings().data_dir) / _KEY_FILE
    if path.exists():
        key = path.read_bytes().strip()
        if key:
            return key
    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    log.info("settings-secret key generated", path=str(path))
    return key


def _box() -> Fernet:
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_resolve_key())
    return _fernet


def encrypt_secret(value: str) -> str:
    """Encrypt a non-empty secret for DB storage; empty stays empty."""
    if not value:
        return value
    return SECRET_PREFIX + _box().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    """Decrypt an ``enc:v1:``-prefixed stored value back to plaintext."""
    return _box().decrypt(value[len(SECRET_PREFIX) :].encode()).decode()


def decrypt_stored(value: object) -> object:
    """Field validator helper: decrypt prefixed strings, pass everything else
    (plaintext secrets from updates, SecretStrs, legacy values) through. A
    value that cannot be decrypted — key rotated or lost — degrades to empty
    with a warning rather than breaking the whole settings load."""
    if isinstance(value, str) and value.startswith(SECRET_PREFIX):
        try:
            return decrypt_secret(value)
        except InvalidToken:
            log.warning("stored secret could not be decrypted — clearing it")
            return ""
    return value
