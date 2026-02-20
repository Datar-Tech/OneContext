"""Reusable DataTable behavior for 'open on Enter / double-click'."""

from __future__ import annotations

from textual import events
from textual.message import Message
from textual.widgets import DataTable


class OpenableDataTable(DataTable):
    """DataTable with a distinct 'activate' action (Enter / double-click).

    - Single click: select the row (RowSelected) and highlight (RowHighlighted).
    - Double click: also emits RowActivated (for opening details).
    - Enter: emits RowActivated (for opening details).
    """

    class RowActivated(Message, bubble=True):
        def __init__(self, data_table: "OpenableDataTable", cursor_row: int, row_key) -> None:
            super().__init__()
            self.data_table = data_table
            self.cursor_row = cursor_row
            self.row_key = row_key

    def action_select_cursor(self) -> None:
        """Activate the current row/cell (Enter)."""
        super().action_select_cursor()
        try:
            if not self.show_cursor or self.cursor_type != "row" or self.row_count == 0:
                return
            row_key = self.coordinate_to_cell_key(self.cursor_coordinate)[0]
            row_index = self.cursor_row
            self.post_message(OpenableDataTable.RowActivated(self, row_index, row_key))
        except Exception:
            return

    async def _on_click(self, event: events.Click) -> None:
        # Preserve default single-click behavior (selects the row/cell).
        await super()._on_click(event)

        # Open on double-click.
        if event.chain < 2:
            return
        try:
            if not self.show_cursor or self.cursor_type != "row" or self.row_count == 0:
                return
            row_key = self.coordinate_to_cell_key(self.cursor_coordinate)[0]
            row_index = self.cursor_row
            self.post_message(OpenableDataTable.RowActivated(self, row_index, row_key))
        except Exception:
            return
