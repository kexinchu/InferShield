import unittest
from pathlib import Path

from sglang.srt.managers.private_service.privacy_detector_custom import PrivacyDetector


CONFIG = Path(__file__).resolve().parents[2] / (
    "python/sglang/srt/managers/private_service/privacy_patterns_config.json"
)


class TestTier1HighPrecision(unittest.TestCase):
    def setUp(self):
        self.det = PrivacyDetector(config_file=str(CONFIG))

    def test_flags_structured_identifiers(self):
        self.assertTrue(self.det.detect_privacy("Contact jane@corp.com").is_private)
        self.assertTrue(self.det.detect_privacy("Call (415) 555-0199 now").is_private)
        self.assertTrue(self.det.detect_privacy("SSN 123-45-6789").is_private)
        self.assertTrue(
            self.det.detect_privacy("passport number XJ32981921").is_private
        )
        self.assertTrue(self.det.detect_privacy("I live at 742 Evergreen Terrace").is_private)

    def test_does_not_flag_public_prose(self):
        news = (
            "Shares rose after the company reported record profit and revenue "
            "in its latest financial statement."
        )
        wiki = "Paris is the capital of France and a major European city."
        chat = "Can you watch this video and search for a better explanation?"
        self.assertFalse(self.det.detect_privacy(news).is_private)
        self.assertFalse(self.det.detect_privacy(wiki).is_private)
        self.assertFalse(self.det.detect_privacy(chat).is_private)

    def test_trie_requires_word_boundary(self):
        self.assertFalse(self.det.detect_privacy("researchers published results").is_private)


if __name__ == "__main__":
    unittest.main()
