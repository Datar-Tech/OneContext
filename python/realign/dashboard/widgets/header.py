"""Aline Dashboard Header Widget with ASCII Logo."""

from textual.widgets import Static

from ..branding import BRANDING


class AlineHeader(Static):
    """Header widget displaying Aline ASCII logo."""

    DEFAULT_CSS = """
    AlineHeader {
        dock: top;
        min-height: 7;
        padding: 1 2;
        color: $text;
    }
    """

    def compose(self):
        """No children - we render directly."""
        return []

    def render(self) -> str:
        """Render the header content."""
        lines = list(BRANDING.header_lines) if BRANDING.header_lines else ["{product_name}"]
        formatted_lines: list[str] = []
        for line in lines:
            try:
                formatted_lines.append(line.format(product_name=BRANDING.product_name))
            except Exception:
                formatted_lines.append(line)
        return "\n".join(formatted_lines)
