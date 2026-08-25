import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.principal_binding import (
    PrincipalAuthenticationError,
    PrincipalBinding,
    bind_openai_user_id,
)
from sglang.srt.server_args import prepare_server_args


class TestPrincipalBinding(unittest.TestCase):
    def test_loads_mapping_and_authenticates_without_exposing_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "principals.json"
            path.write_text(
                json.dumps({"opaque-secret-token": "trusted-victim"}),
                encoding="utf-8",
            )
            binding = PrincipalBinding.from_json_file(str(path))

        self.assertEqual(
            binding.authenticate("Bearer opaque-secret-token"), "trusted-victim"
        )

        with self.assertRaises(PrincipalAuthenticationError) as context:
            binding.authenticate("Bearer unknown-secret-token")
        self.assertNotIn("unknown-secret-token", str(context.exception))

    def test_missing_and_unknown_bearers_are_rejected(self):
        binding = PrincipalBinding({"known": "principal-a"})

        for authorization in (None, "", "known", "Basic known", "Bearer unknown"):
            with self.subTest(authorization=authorization):
                with self.assertRaises(PrincipalAuthenticationError):
                    binding.authenticate(authorization)

    def test_trusted_principal_overwrites_native_user_id(self):
        binding = PrincipalBinding({"attacker-token": "trusted-attacker"})
        principal = binding.authenticate("Bearer attacker-token")

        single = {"user_id": "spoofed-victim", "max_new_tokens": 1}
        self.assertIs(binding.bind_sampling_params(single, principal), single)
        self.assertEqual(single["user_id"], "trusted-attacker")

        batch = [{"user_id": "victim"}, {"temperature": 0.0}]
        binding.bind_sampling_params(batch, principal)
        self.assertEqual(
            [params["user_id"] for params in batch],
            ["trusted-attacker", "trusted-attacker"],
        )

    def test_trusted_principal_overwrites_openai_user_id(self):
        request = SimpleNamespace(user_id="spoofed-victim")
        bind_openai_user_id(request, "trusted-attacker")
        self.assertEqual(request.user_id, "trusted-attacker")

        bind_openai_user_id(request, None)
        self.assertEqual(request.user_id, "trusted-attacker")

    def test_disabled_binding_preserves_legacy_behavior(self):
        binding = PrincipalBinding()
        params = {"user_id": "client-controlled"}

        self.assertIsNone(binding.authenticate(None))
        self.assertEqual(params["user_id"], "client-controlled")

    def test_server_argument_is_explicit_opt_in(self):
        disabled = prepare_server_args(["--model-path", "model"])
        enabled = prepare_server_args(
            [
                "--model-path",
                "model",
                "--principal-binding-file",
                "/tmp/principals.json",
            ]
        )

        self.assertIsNone(disabled.principal_binding_file)
        self.assertEqual(enabled.principal_binding_file, "/tmp/principals.json")


if __name__ == "__main__":
    unittest.main()
