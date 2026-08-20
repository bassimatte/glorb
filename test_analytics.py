import re
import unittest
from pathlib import Path


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = Path("templates/index.html").read_text(encoding="utf-8")
        cls.docs = Path("docs/index.html").read_text(encoding="utf-8")

    def test_analytics_is_limited_to_canonical_glorb_path(self):
        for html in (self.template, self.docs):
            self.assertIn("const GLORB_ANALYTICS_HOST = 'bassimatte.github.io';", html)
            self.assertIn("const GLORB_ANALYTICS_PATH = '/glorb';", html)
            self.assertIn("location.pathname.startsWith(`${GLORB_ANALYTICS_PATH}/`)", html)
            self.assertIn("script.dataset.domains = GLORB_ANALYTICS_HOST;", html)

    def test_shared_website_uses_glorb_tag_and_event_prefix(self):
        for html in (self.template, self.docs):
            self.assertIn(
                "const GLORB_UMAMI_WEBSITE_ID = 'b2300e0b-bc69-49d7-ad05-06f980b5ed38';",
                html,
            )
            self.assertIn("const GLORB_ANALYTICS_TAG = 'glorb';", html)
            self.assertIn("const GLORB_ANALYTICS_EVENT_PREFIX = 'glorb_';", html)
            self.assertIn("script.dataset.tag = GLORB_ANALYTICS_TAG;", html)
            self.assertIn(
                "window.umami.track(`${GLORB_ANALYTICS_EVENT_PREFIX}${eventName}`, properties);",
                html,
            )
            self.assertNotIn("window.umami.track(eventName, properties);", html)

    def test_privacy_controls_are_enabled(self):
        for html in (self.template, self.docs):
            self.assertIn("script.dataset.excludeSearch = 'true';", html)
            self.assertIn("script.dataset.excludeHash = 'true';", html)
            self.assertIn("script.dataset.doNotTrack = 'true';", html)
            self.assertIn("if (!analyticsIsConfigured()) return false;", html)

    def test_events_and_properties_are_allowlisted(self):
        for html in (self.template, self.docs):
            schema = re.search(
                r"const GLORB_ANALYTICS_SCHEMA = Object\.freeze\(\{(.*?)\n  \}\);",
                html,
                re.DOTALL,
            )
            self.assertIsNotNone(schema)
            for event in (
                "generation_completed",
                "generation_failed",
                "audio_started",
                "download_completed",
                "share_copied",
            ):
                self.assertIn(f"{event}:", schema.group(1))
                self.assertRegex(html, rf"trackUsage\('{event}'")
            self.assertIn("if (allowed.includes(value)) props[key] = value;", html)

    def test_sensitive_values_are_not_sent(self):
        for html in (self.template, self.docs):
            calls = re.findall(
                r"trackUsage\('([^']+)'\s*,\s*\{(.*?)\}\);", html, re.DOTALL
            )
            self.assertTrue(calls)
            properties = "\n".join(props for _, props in calls)
            for forbidden in (
                "energy:", "brightness:", "chaos:", "url:", "filename:",
                "audio:", "error:", "message:", "query:",
            ):
                self.assertNotIn(forbidden, properties)

    def test_privacy_notice_is_visible(self):
        for html in (self.template, self.docs):
            self.assertIn("anonymous, aggregate usage statistics", html)
            self.assertIn("Analytics is disabled for local installations", html)


if __name__ == "__main__":
    unittest.main()
