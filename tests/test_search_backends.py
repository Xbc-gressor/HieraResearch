"""Jina key rotation: auth/quota rejections fail over to backup keys."""

from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import search_backends  # noqa: E402


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://s.jina.ai/", code, "err", {}, None)


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class JinaKeyRotationTests(unittest.TestCase):
    def setUp(self) -> None:
        pool = search_backends.JinaKeyPool()
        patcher = mock.patch.object(search_backends, "JINA_KEYS", pool)
        self.addCleanup(patcher.stop)
        self.pool = patcher.start()
        env = mock.patch.dict(os.environ, {
            search_backends.JINA_KEY_ENV: "key-a",
            search_backends.JINA_KEY_BACKUPS_ENV: "key-b",
        })
        self.addCleanup(env.stop)
        env.start()

    def _search_once(self, responder) -> tuple[dict, list[str | None]]:
        auths: list[str | None] = []

        def fake_urlopen(request, timeout=None):
            auths.append(request.headers.get("Authorization"))
            return responder(len(auths))

        with mock.patch.object(search_backends.urllib.request, "urlopen",
                               fake_urlopen):
            result = search_backends.JinaSearchBackend()._search_sync("q", 5)
        return result, auths

    def test_quota_rejection_rotates_to_the_backup_key(self) -> None:
        payload = {"data": [{"url": "https://x.example", "title": "t",
                             "description": "d"}]}

        def responder(call_no: int):
            if call_no == 1:
                raise _http_error(402)
            return _Resp(payload)

        result, auths = self._search_once(responder)
        self.assertEqual(auths, ["Bearer key-a", "Bearer key-b"])
        self.assertEqual(result["items"][0]["url"], "https://x.example")
        self.assertEqual(self.pool.index, 1)  # dead key stays skipped

    def test_rotation_exhaustion_reraises(self) -> None:
        def responder(call_no: int):
            raise _http_error(402)

        with self.assertRaises(urllib.error.HTTPError):
            self._search_once(responder)
        self.assertEqual(self.pool.index, 1)

    def test_transient_errors_do_not_rotate(self) -> None:
        def responder(call_no: int):
            raise _http_error(500)

        with self.assertRaises(urllib.error.HTTPError):
            self._search_once(responder)
        self.assertEqual(self.pool.index, 0)


if __name__ == "__main__":
    unittest.main()
