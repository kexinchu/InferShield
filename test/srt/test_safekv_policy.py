import threading
import time
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from sglang.srt.mem_cache.safekv_policy import (
    DurableLedger,
    NamespaceKey,
    PublicRegistry,
    SafeKVMetrics,
)


class TestDurableLedger(unittest.TestCase):
    def test_concurrent_instances_share_one_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ledger.json")
            ledgers = [DurableLedger(path) for _ in range(4)]

            def reserve(index: int) -> bool:
                accepted, _, _ = ledgers[index % len(ledgers)].reserve_hit(
                    "fingerprint", 10
                )
                return accepted

            with ThreadPoolExecutor(max_workers=32) as pool:
                accepted = sum(pool.map(reserve, range(128)))

            self.assertEqual(accepted, 10)
            self.assertEqual(
                DurableLedger(path).charged_hits("fingerprint"), 10
            )

    def test_corrupt_and_unavailable_ledgers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            path.write_text("{bad-json", encoding="utf-8")
            corrupt = DurableLedger(str(path))
            self.assertFalse(corrupt.is_operational)
            self.assertEqual(
                corrupt.reserve_hit("fingerprint", 10)[2],
                "recovery_incomplete",
            )

            available = DurableLedger()
            available.set_available(False)
            self.assertEqual(
                available.reserve_hit("fingerprint", 10)[2], "unavailable"
            )


class TestPublicRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = [101, 42, 102]
        self.registry = PublicRegistry(b"operator-secret", policy_epoch=7)
        self.authorization = self.registry.issue(
            public_object_id="public-1",
            issuer="operator",
            model_id="model-a",
            tokenizer_version="tokenizer-v1",
            token_ids=self.tokens,
            expires_at=time.time() + 60,
        )

    def verify(self, authorization=None, **overrides):
        arguments = {
            "authorization": authorization or self.authorization,
            "model_id": "model-a",
            "tokenizer_version": "tokenizer-v1",
            "token_ids": self.tokens,
        }
        arguments.update(overrides)
        return self.registry.verify(**arguments)

    def test_valid_authorization(self) -> None:
        result = self.verify()
        self.assertTrue(result)
        self.assertEqual(result.reason, PublicRegistry.OK)

    def test_wrong_key_forged_mac_is_rejected_and_not_installed(self) -> None:
        attacker = PublicRegistry(b"attacker-key", policy_epoch=7)
        forged = attacker.issue(
            "forged-object",
            "attacker",
            "model-a",
            "tokenizer-v1",
            self.tokens,
            time.time() + 60,
        )

        result = self.registry.install(forged, self.tokens)

        self.assertFalse(result)
        self.assertEqual(result.reason, PublicRegistry.INVALID_MAC)
        self.assertNotIn(
            "forged-object", self.registry.snapshot()["authorizations"]
        )

    def test_wrong_model_tokenizer_and_length(self) -> None:
        self.assertEqual(
            self.verify(model_id="model-b").reason,
            PublicRegistry.WRONG_MODEL,
        )
        self.assertEqual(
            self.verify(tokenizer_version="tokenizer-v2").reason,
            PublicRegistry.WRONG_TOKENIZER,
        )
        self.assertEqual(
            self.verify(token_ids=self.tokens[:-1]).reason,
            PublicRegistry.WRONG_LENGTH,
        )

    def test_stale_epoch(self) -> None:
        self.registry.reset(policy_epoch=8)
        self.assertEqual(self.verify().reason, PublicRegistry.STALE_EPOCH)

    def test_expired(self) -> None:
        result = self.verify(now=self.authorization.expires_at)
        self.assertEqual(result.reason, PublicRegistry.EXPIRED)

    def test_revoked(self) -> None:
        self.assertTrue(self.registry.revoke("public-1"))
        self.assertEqual(self.verify().reason, PublicRegistry.REVOKED)

    def test_revoked_flag_cannot_be_installed(self) -> None:
        revoked = replace(self.authorization, revoked=True)
        result = self.registry.install(revoked, self.tokens)
        self.assertFalse(result)
        self.assertNotEqual(result.reason, PublicRegistry.OK)

    def test_concurrent_install_and_revoke_remains_revoked(self) -> None:
        destination = PublicRegistry(b"operator-secret", policy_epoch=7)
        start = threading.Barrier(3)

        def install_many() -> None:
            start.wait()
            for _ in range(500):
                destination.install(self.authorization, self.tokens)

        def revoke_many() -> None:
            start.wait()
            for _ in range(500):
                destination.revoke(self.authorization.public_object_id)

        installers = threading.Thread(target=install_many)
        revokers = threading.Thread(target=revoke_many)
        installers.start()
        revokers.start()
        start.wait()
        installers.join()
        revokers.join()

        result = destination.verify(
            self.authorization,
            "model-a",
            "tokenizer-v1",
            self.tokens,
        )
        self.assertEqual(result.reason, PublicRegistry.REVOKED)


class TestNamespaceAndMetrics(unittest.TestCase):
    def test_private_and_public_namespace_keys_are_distinct(self) -> None:
        private = NamespaceKey.Private("same-id", "same-fingerprint")
        public = NamespaceKey.Public("same-id", "same-fingerprint")
        self.assertNotEqual(private, public)
        self.assertEqual(
            private, NamespaceKey.private("same-id", "same-fingerprint")
        )

    def test_metrics_snapshot_and_reset(self) -> None:
        metrics = SafeKVMetrics()
        metrics.increment(
            "unauth_public_promotions", principal="tenant-a"
        )
        metrics.increment("public_object_created", 2)
        metrics.record_event("authorization_rejected", reason="invalid_mac")

        snapshot = metrics.snapshot()
        self.assertEqual(
            snapshot["counters"]["unauth_public_promotions"], 1
        )
        self.assertEqual(snapshot["counters"]["public_object_created"], 2)
        self.assertEqual(len(snapshot["events"]), 2)

        metrics.reset()
        reset_snapshot = metrics.snapshot()
        self.assertTrue(
            all(value == 0 for value in reset_snapshot["counters"].values())
        )
        self.assertEqual(reset_snapshot["events"], ())
        self.assertEqual(snapshot["counters"]["public_object_created"], 2)


if __name__ == "__main__":
    unittest.main()
