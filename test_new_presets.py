import re
import unittest
from pathlib import Path

import numpy as np

import generate
import main


NEW_PRESETS = ("glass", "clockwork", "creature", "electricity", "cave")


class NewPresetTests(unittest.TestCase):
    def test_catalogs_and_routes_include_every_new_preset(self):
        server = Path("server.py").read_text(encoding="utf-8")
        template = Path("templates/index.html").read_text(encoding="utf-8")
        docs = Path("docs/index.html").read_text(encoding="utf-8")
        for preset in NEW_PRESETS:
            self.assertIn(preset, main.SOUND_PRESETS)
            self.assertIn(preset, main._PRESET_EVENTS)
            self.assertIn(preset, generate.MODES)
            self.assertIn(f'"{preset}":', server)
            self.assertIn(f'data-mode="{preset}"', template)
            self.assertIn(f'data-mode="{preset}"', docs)

    def test_web_catalog_matches_python_catalog(self):
        html = Path("templates/index.html").read_text(encoding="utf-8")
        preset_section = html[html.index('id="mode-pills"'):html.index('id="nature-row"')]
        web_presets = re.findall(r'data-mode="([^"]+)"', preset_section)
        self.assertEqual(web_presets, list(main.SOUND_PRESETS))

    def test_continuous_arp_and_groove_audio_is_valid(self):
        np.random.seed(7)
        for preset in NEW_PRESETS:
            renderers = (
                getattr(main, f"make_{preset}_sequence")(.15),
                main.make_arp_sequence(.15, preset=preset),
                main.make_groove_sequence(.15, preset=preset),
            )
            for audio in renderers:
                self.assertEqual(audio.shape, (int(.15 * main.SAMPLE_RATE), 2))
                self.assertTrue(np.isfinite(audio).all())
                peak = float(np.max(np.abs(audio)))
                self.assertGreater(peak, 0)
                self.assertLessEqual(peak, .989)


if __name__ == "__main__":
    unittest.main()
