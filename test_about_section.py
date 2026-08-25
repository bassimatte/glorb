import re
import unittest
from pathlib import Path


class AboutSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = (
            Path("templates/index.html").read_text(encoding="utf-8"),
            Path("docs/index.html").read_text(encoding="utf-8"),
        )

    def test_about_section_is_visible_and_semantic(self):
        for html in self.pages:
            shell_end = html.index("</div><!-- .shell -->")
            about_start = html.index('<section class="about-glorb"')
            modal_start = html.index("<!-- ═══════════ INFO MODAL")
            self.assertLess(shell_end, about_start)
            self.assertLess(about_start, modal_start)
            self.assertIn('aria-labelledby="about-glorb-title"', html)
            self.assertIn('<h2 id="about-glorb-title">', html)

    def test_about_section_has_explanation_and_project_links(self):
        for html in self.pages:
            section = re.search(
                r'<section class="about-glorb".*?</section>', html, re.DOTALL
            )
            self.assertIsNotNone(section)
            content = section.group(0)
            for heading in (
                "What Glorb creates",
                "How it works",
                "What you can use it for",
            ):
                self.assertIn(heading, content)
            self.assertIn("https://freesound.org/people/bassimat/packs/45732/", content)
            self.assertIn("https://github.com/bassimatte/glorb", content)
            self.assertIn("https://bassimatte.github.io/#instruments", content)


if __name__ == "__main__":
    unittest.main()
