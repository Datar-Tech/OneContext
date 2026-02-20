"""Branding constants for user-visible dashboard copy.

Keep CLI/internal identifiers (e.g. `aline`, `~/.aline`, env vars) unchanged for compatibility.
Only use these constants for user-visible strings in the dashboard UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardBranding:
    product_name: str = "OneContext"

    # Header ASCII art (one string per line). This defaults to the original Aline
    # "block/box-drawing" style. The header widget appends version info to the last line.
    header_lines: tuple[str, ...] = (
        " ████╗ █╗   █╗█████╗ ████╗ ████╗ █╗   █╗█████╗█████╗█╗  █╗█████╗",
        "█╔═══█╗██╗  █║█╔═══╝█╔═══╝█╔═══█╗██╗  █║╚═█╔═╝█╔═══╝╚█╗█╔╝╚═█╔═╝",
        "█║   █║█╔█╗ █║████╗ █║    █║   █║█╔█╗ █║  █║  ████╗  ╚█╔╝   █║  ",
        "█║   █║█║╚█╗█║█╔══╝ █║    █║   █║█║╚█╗█║  █║  █╔══╝  █╔█╗   █║  ",
        "╚████╔╝█║ ╚██║█████╗╚████╗╚████╔╝█║ ╚██║  █║  █████╗█╔╝ █╗  █║  ",
        # " ██████╗  ██████╗ ███╗   ██╗████████╗███████╗██╗  ██╗████████╗",
        # "██╔════╝ ██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝",
        # "██║      ██║   ██║██╔██╗ ██║   ██║   █████╗   ╚███╔╝    ██║   ",
        # "██║      ██║   ██║██║╚██╗██║   ██║   ██╔══╝   ██╔██╗    ██║   ",
        # "╚██████╗ ╚██████╔╝██║ ╚████║   ██║   ███████╗██╔╝ ██╗   ██║   ",
    )

    @property
    def dashboard_title(self) -> str:
        return f"{self.product_name} Dashboard"

    @property
    def dashboard_label(self) -> str:
        return f"{self.product_name} dashboard"

    @property
    def doctor_label(self) -> str:
        return f"{self.product_name} Doctor"

    @property
    def import_page_title(self) -> str:
        return f"{self.product_name} Import"


BRANDING = DashboardBranding()
