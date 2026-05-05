"""Testes unitários de src/history.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from src.history import HistoryEntry, append_entry, _MAX_HISTORY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(query: str = "consulta", files: list[str] | None = None) -> HistoryEntry:
    return HistoryEntry(
        query=query,
        files=files or ["planta.dxf"],
        successful=1,
        failed=0,
        batch_result=MagicMock(),
        template="ambientes_e_areas",
        timestamp=datetime(2026, 5, 5, 14, 30),
    )


# ---------------------------------------------------------------------------
# Propriedades de exibição
# ---------------------------------------------------------------------------

class TestHistoryEntryDisplay:
    def test_time_label(self):
        e = _entry()
        assert e.time_label == "14:30"

    def test_date_label(self):
        e = _entry()
        assert e.date_label == "05/05 14:30"

    def test_query_preview_short(self):
        e = _entry("Curta")
        assert e.query_preview == "Curta"

    def test_query_preview_truncated(self):
        e = _entry("A" * 100)
        assert e.query_preview.endswith("…")
        assert len(e.query_preview) <= 56  # 55 chars + ellipsis

    def test_files_preview_short(self):
        e = _entry(files=["a.dxf", "b.pdf"])
        assert "a.dxf" in e.files_preview
        assert "b.pdf" in e.files_preview

    def test_files_preview_truncated(self):
        e = _entry(files=["a.dxf", "b.pdf", "c.dxf", "d.pdf", "e.dxf"])
        assert "+2" in e.files_preview

    def test_status_summary_all_ok(self):
        e = HistoryEntry(
            query="q", files=["f.dxf"], successful=3, failed=0,
            batch_result=MagicMock(),
        )
        assert "✅" in e.status_summary
        assert "3" in e.status_summary

    def test_status_summary_with_errors(self):
        e = HistoryEntry(
            query="q", files=["f.dxf"], successful=2, failed=1,
            batch_result=MagicMock(),
        )
        assert "❌" in e.status_summary
        assert "1" in e.status_summary

    def test_unique_ids(self):
        ids = {_entry().id for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# append_entry
# ---------------------------------------------------------------------------

class TestAppendEntry:
    def test_appends_single_entry(self):
        history: list[HistoryEntry] = []
        append_entry(history, _entry("q1"))
        assert len(history) == 1

    def test_appends_multiple_entries(self):
        history: list[HistoryEntry] = []
        for i in range(5):
            append_entry(history, _entry(f"consulta {i}"))
        assert len(history) == 5

    def test_respects_max_history(self):
        history: list[HistoryEntry] = []
        for i in range(_MAX_HISTORY + 5):
            append_entry(history, _entry(f"consulta {i}"))
        assert len(history) == _MAX_HISTORY

    def test_oldest_entry_removed_when_full(self):
        history: list[HistoryEntry] = []
        for i in range(_MAX_HISTORY):
            append_entry(history, _entry(f"consulta {i}"))
        append_entry(history, _entry("nova consulta"))
        assert history[0].query == "consulta 1"
        assert history[-1].query == "nova consulta"

    def test_deduplicates_immediate_repeat(self):
        """Duplo clique no botão não deve criar entrada duplicada."""
        history: list[HistoryEntry] = []
        e1 = _entry("mesma consulta", ["a.dxf"])
        e2 = _entry("mesma consulta", ["a.dxf"])
        append_entry(history, e1)
        append_entry(history, e2)
        assert len(history) == 1
        # deve manter a versão mais recente
        assert history[0] is e2

    def test_different_query_not_deduplicated(self):
        history: list[HistoryEntry] = []
        append_entry(history, _entry("consulta A"))
        append_entry(history, _entry("consulta B"))
        assert len(history) == 2

    def test_same_query_different_files_not_deduplicated(self):
        history: list[HistoryEntry] = []
        append_entry(history, _entry("consulta", ["a.dxf"]))
        append_entry(history, _entry("consulta", ["b.pdf"]))
        assert len(history) == 2
