"""AbstractCore web console (served by ``abstractcore serve`` at ``/console``).

``render_console_html()`` returns the standalone page; ``fragment(kind)``
returns the self-contained Models / Engines tab bodies that
abstractgateway embeds in its own console. See ``docs/console.md``.
"""

from .web import FRAGMENT_KINDS, fragment, render_console_html

__all__ = ["FRAGMENT_KINDS", "fragment", "render_console_html"]
