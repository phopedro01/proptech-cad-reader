"""
Testes unitários do BatchProcessor.
Usa DXFs sintéticos (ezdxf) e PDFs sintéticos (PyMuPDF) — sem arquivos reais.
"""

from __future__ import annotations

import math
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import ezdxf
import fitz
import pytest

from src.batch.batch_processor import BatchProcessor, BatchResult, FileJob


# ---------------------------------------------------------------------------
# Helpers para criar bytes de arquivos sintéticos
# ---------------------------------------------------------------------------

def _dxf_bytes(texts: list[str] | None = None) -> bytes:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for t in (texts or ["SALA DE ESTAR 20m²"]):
        msp.add_text(t, dxfattribs={"layer": "A-TEXT", "height": 0.25,
                                     "insert": (0, 0, 0)})
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        path = f.name
    try:
        doc.saveas(path)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def _pdf_bytes(text: str = "Dormitório 01 - 12.30m²") -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), text, fontsize=12)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Testes do FileJob
# ---------------------------------------------------------------------------

class TestFileJob:
    def test_is_ready_for_llm(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=100,
                    status="ready", raw_summary="texto")
        assert j.is_ready_for_llm is True

    def test_not_ready_if_no_summary(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=100,
                    status="ready", raw_summary=None)
        assert j.is_ready_for_llm is False

    def test_not_ready_if_pending(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=100,
                    status="pending", raw_summary="texto")
        assert j.is_ready_for_llm is False

    def test_size_display_kb(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=1500)
        assert "KB" in j.size_display

    def test_size_display_mb(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=2_000_000)
        assert "MB" in j.size_display

    def test_type_badge_dxf(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=0, file_type="dxf")
        assert "DXF" in j.type_badge

    def test_type_badge_pdf(self):
        j = FileJob(filename="f.pdf", file_hash="abc", size_bytes=0, file_type="pdf")
        assert "PDF" in j.type_badge

    def test_total_duration(self):
        j = FileJob(filename="f.dxf", file_hash="abc", size_bytes=0,
                    parse_duration_s=1.2, llm_duration_s=3.5)
        assert math.isclose(j.total_duration_s, 4.7)


# ---------------------------------------------------------------------------
# Testes do parse_single
# ---------------------------------------------------------------------------

class TestParseSingle:
    def test_parse_dxf_ok(self):
        data = _dxf_bytes()
        bp = BatchProcessor()
        job = bp.parse_single("planta.dxf", data, "hash1")
        assert job.status == "ready"
        assert job.file_type == "dxf"
        assert job.raw_summary is not None
        assert len(job.raw_summary) > 0
        assert job.parse_duration_s > 0

    def test_parse_pdf_ok(self):
        data = _pdf_bytes()
        bp = BatchProcessor()
        job = bp.parse_single("memorial.pdf", data, "hash2")
        assert job.status == "ready"
        assert job.file_type == "pdf"
        assert job.raw_summary is not None

    def test_parse_unsupported_format(self):
        bp = BatchProcessor()
        job = bp.parse_single("arquivo.dwg", b"fake dwg content", "hash3")
        assert job.status == "error"
        assert job.error is not None
        assert "suportado" in job.error.lower() or "DWG" in job.error

    def test_parse_corrupted_dxf(self):
        bp = BatchProcessor()
        job = bp.parse_single("corrupto.dxf", b"not a dxf %%%", "hash4")
        assert job.status == "error"

    def test_parse_records_duration(self):
        data = _dxf_bytes()
        bp = BatchProcessor()
        job = bp.parse_single("planta.dxf", data, "hash5")
        assert job.parse_duration_s >= 0.0

    def test_parse_with_layer_filter(self):
        data = _dxf_bytes(["Texto na layer A-TEXT"])
        bp = BatchProcessor(dxf_layers=["A-TEXT"])
        job = bp.parse_single("planta.dxf", data, "hash6")
        assert job.status == "ready"


# ---------------------------------------------------------------------------
# Testes do run_llm_batch
# ---------------------------------------------------------------------------

class TestRunLlmBatch:
    def _make_ready_job(self, filename: str = "f.dxf") -> FileJob:
        j = FileJob(
            filename=filename, file_hash="h", size_bytes=100,
            status="ready", file_type="dxf",
            raw_summary="=== TEXTOS ===\n[TEXT] SALA DE ESTAR",
        )
        return j

    def test_skips_non_ready_jobs(self):
        """Jobs pendentes não devem ser afetados."""
        pending = FileJob(filename="p.dxf", file_hash="h0", size_bytes=0,
                          status="pending")
        ready = self._make_ready_job("r.dxf")

        with patch("src.batch.batch_processor.NLPProcessor") as mock_cls:
            mock_cls.return_value.process.return_value = {"ambientes": []}
            bp = BatchProcessor()
            result = bp.run_llm_batch([pending, ready], "consulta teste")

        assert pending.status == "pending"
        assert ready.status == "done"

    def test_successful_llm_sets_done(self):
        job = self._make_ready_job()
        with patch("src.batch.batch_processor.NLPProcessor") as mock_cls:
            mock_cls.return_value.process.return_value = {
                "ambientes": [{"nome": "Sala", "area_m2": 20}]
            }
            bp = BatchProcessor()
            result = bp.run_llm_batch([job], "extraia ambientes")

        assert job.status == "done"
        assert job.llm_result is not None
        assert "ambientes" in job.llm_result

    def test_llm_error_sets_error_status(self):
        job = self._make_ready_job()
        with patch("src.batch.batch_processor.NLPProcessor") as mock_cls:
            mock_cls.return_value.process.side_effect = Exception("API timeout")
            bp = BatchProcessor()
            result = bp.run_llm_batch([job], "qualquer coisa")

        assert job.status == "error"
        assert "LLM" in job.error

    def test_one_error_does_not_stop_batch(self):
        """Erro em um arquivo não deve impedir processamento dos demais."""
        j1 = self._make_ready_job("ok.dxf")
        j2 = self._make_ready_job("fail.dxf")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("falha simulada")
            return {"ambientes": []}

        with patch("src.batch.batch_processor.NLPProcessor") as mock_cls:
            mock_cls.return_value.process.side_effect = side_effect
            bp = BatchProcessor()
            result = bp.run_llm_batch([j1, j2], "consulta")

        assert j1.status == "error"
        assert j2.status == "done"

    def test_progress_callback_called(self):
        job = self._make_ready_job()
        calls: list[tuple] = []

        def on_progress(current, total, filename):
            calls.append((current, total, filename))

        with patch("src.batch.batch_processor.NLPProcessor") as mock_cls:
            mock_cls.return_value.process.return_value = {}
            bp = BatchProcessor()
            bp.run_llm_batch([job], "q", on_progress=on_progress)

        assert len(calls) >= 2
        # primeira chamada: before file
        assert calls[0][0] == 0
        # última chamada: completion signal (filename == '')
        assert calls[-1][2] == ""

    def test_empty_jobs_returns_empty_result(self):
        bp = BatchProcessor()
        result = bp.run_llm_batch([], "consulta")
        assert result.total == 0
        assert result.successful == []


# ---------------------------------------------------------------------------
# Testes do BatchResult
# ---------------------------------------------------------------------------

class TestBatchResult:
    def _make_result(self) -> BatchResult:
        j1 = FileJob(filename="a.dxf", file_hash="h1", size_bytes=0,
                     status="done", file_type="dxf",
                     llm_result={"ambientes": [{"nome": "Sala", "area_m2": 20.0}]})
        j2 = FileJob(filename="b.dxf", file_hash="h2", size_bytes=0,
                     status="done", file_type="dxf",
                     llm_result={"ambientes": [{"nome": "Quarto", "area_m2": 12.0}]})
        j3 = FileJob(filename="c.dxf", file_hash="h3", size_bytes=0,
                     status="error", error="falha")
        return BatchResult(jobs=[j1, j2, j3], query="ambientes e áreas")

    def test_successful_count(self):
        r = self._make_result()
        assert len(r.successful) == 2

    def test_failed_count(self):
        r = self._make_result()
        assert len(r.failed) == 1

    def test_success_rate(self):
        r = self._make_result()
        assert math.isclose(r.success_rate, 2 / 3)

    def test_combined_tables_merges_rows(self):
        r = self._make_result()
        tables = r.combined_tables()
        assert "ambientes" in tables
        df = tables["ambientes"]
        assert len(df) == 2  # 1 ambiente por arquivo
        assert "Arquivo" in df.columns

    def test_combined_tables_has_source_column(self):
        r = self._make_result()
        df = r.combined_tables()["ambientes"]
        sources = set(df["Arquivo"].tolist())
        assert "a" in sources
        assert "b" in sources

    def test_to_dict_structure(self):
        r = self._make_result()
        d = r.to_dict()
        assert "query" in d
        assert "summary" in d
        assert "results" in d
        assert d["summary"]["total"] == 3
        assert d["summary"]["successful"] == 2

    def test_to_dict_has_timings(self):
        r = self._make_result()
        first = r.to_dict()["results"][0]
        assert "timings_s" in first
        assert "parse" in first["timings_s"]
