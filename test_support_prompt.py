import re
import unittest
from pathlib import Path


class SupportPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = (
            Path("templates/index.html").read_text(encoding="utf-8"),
            Path("docs/index.html").read_text(encoding="utf-8"),
        )

    def test_support_links_use_shared_page(self):
        for html in self.pages:
            self.assertIn(
                "const GLORB_SUPPORT_URL = 'https://bassimatte.github.io/support/';",
                html,
            )
            for element_id in (
                "support-open",
                "support-about-link",
                "support-settings-link",
            ):
                self.assertIn(f'id="{element_id}"', html)
            self.assertIn("document.getElementById(id).href = GLORB_SUPPORT_URL;", html)

    def test_prompt_counts_only_successful_hosted_synthesis(self):
        for html in self.pages:
            self.assertIn("function supportPromptIsHosted()", html)
            self.assertIn("return analyticsIsConfigured();", html)
            self.assertIn("if (!supportPromptIsHosted()) return null;", html)
            self.assertIn("const SUPPORT_MIN_GENERATION_UNITS = 0.15;", html)
            self.assertIn("const SUPPORT_UI_PACK_UNITS = 0.5;", html)
            self.assertIn("addGenerationSupportUnits(buf.duration);", html)
            self.assertIn("addSupportUnits(SUPPORT_UI_PACK_UNITS);", html)
            completed = html.index("trackUsage('generation_completed'")
            units = html.index("addGenerationSupportUnits(buf.duration);", completed)
            failed = html.index("trackUsage('generation_failed'", units)
            self.assertLess(completed, units)
            self.assertLess(units, failed)

    def test_frequency_snooze_and_opt_out_are_conservative(self):
        for html in self.pages:
            for declaration in (
                "const SUPPORT_INITIAL_UNITS = 15;",
                "const SUPPORT_SNOOZE_UNITS = 20;",
                "const SUPPORT_SNOOZE_MS = 14 * 24 * 60 * 60 * 1000;",
                "const SUPPORT_MAX_SHOWS_PER_YEAR = 2;",
            ):
                self.assertIn(declaration, html)
            self.assertIn("state.optedOut = true;", html)
            self.assertIn("state.shownAt.length < SUPPORT_MAX_SHOWS_PER_YEAR", html)
            self.assertIn(
                "state.nextUnits = Math.max(state.nextUnits, state.units + SUPPORT_SNOOZE_UNITS);",
                html,
            )

    def test_prompt_is_accessible_and_does_not_interrupt_work(self):
        for html in self.pages:
            self.assertIn('role="dialog" aria-modal="true"', html)
            self.assertIn('id="support-close" aria-label="Close support request"', html)
            self.assertIn('id="support-later">Not now</button>', html)
            self.assertIn('id="support-optout">Don’t ask again</button>', html)
            self.assertIn("if (isPlaying || document.getElementById('btn-generate')?.disabled) return;", html)
            self.assertIn("maybeShowSupportPrompt('listening');", html)
            self.assertIn("if (!isPlaying) maybeShowSupportPrompt('download');", html)

    def test_local_score_is_not_sent_to_umami(self):
        for html in self.pages:
            self.assertIn("const GLORB_SUPPORT_STATE_KEY = 'glorb_support_v1';", html)
            self.assertIn("localStorage.setItem(GLORB_SUPPORT_STATE_KEY", html)
            self.assertIn("stored only in this browser and is never sent to analytics", html)
            analytics_calls = re.findall(
                r"trackUsage\('support_[^']+'\s*,\s*\{(.*?)\}\);",
                html,
                re.DOTALL,
            )
            self.assertTrue(analytics_calls)
            self.assertNotIn("units", "\n".join(analytics_calls))
            self.assertIn("support_prompt_shown: { trigger: ['listening', 'download'] }", html)
            self.assertIn("support_action: { action: ['opened', 'later', 'closed', 'opted_out'] }", html)
            self.assertIn("support_link_opened: { source: ['prompt', 'about', 'settings'] }", html)

    def test_every_support_interaction_has_analytics(self):
        for html in self.pages:
            self.assertIn("trackUsage('support_prompt_shown', { trigger });", html)
            self.assertIn("trackUsage('support_action', { action });", html)
            for source in ("prompt", "about", "settings"):
                self.assertIn(
                    f"trackUsage('support_link_opened', {{ source: '{source}' }});",
                    html,
                )


if __name__ == "__main__":
    unittest.main()
