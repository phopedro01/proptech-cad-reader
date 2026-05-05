"""
Testes unitários do PDFParser.
Gera PDFs sintéticos em memória com PyMuPDF — não requer arquivo externo.
"""

import math
import os
import tempfile

import fitz
import pytest

from src.parsers.pdf_parser import (
    PDFParser,
    PDFParseResult,
    TableData,
    _preprocess_for_ocr,
    _OPENCV_OK,
)


# ---------------------------------------------------------------------------
# Fixtures — PDFs sintéticos gerados em memória
# ---------------------------------------------------------------------------

def _make_digital_pdf(text_per_page: list[str]) -> bytes:
    """
    Cria PDF digital com texto nativo.
    Cada página recebe o texto principal + linhas de preenchimento para
    garantir > _MIN_CHARS_PER_PAGE (80) chars extraídos por PyMuPDF.
    """
    _FILLER = [
        "Projeto: Residência Unifamiliar — Pavimento Térreo",
        "Escala 1:50  |  Responsável: Arq. Silva  |  CAU A12345",
        "Revestimento: consultar memorial descritivo anexo",
    ]
    doc = fitz.open()
    for text in text_per_page:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 100), text, fontsize=12)
        for i, line in enumerate(_FILLER):
            page.insert_text((50, 125 + i * 22), line, fontsize=9)
    return doc.tobytes()


def _make_empty_pdf(n_pages: int = 1) -> bytes:
    """Cria PDF sem texto (simula escaneado)."""
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page(width=595, height=842)
    return doc.tobytes()


def _save_tmp_pdf(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        return f.name


# ---------------------------------------------------------------------------
# Testes de detecção de modo
# ---------------------------------------------------------------------------

class TestScanDetection:
    def test_digital_pdf_not_scanned(self):
        pdf_bytes = _make_digital_pdf(["SALA DE ESTAR - 20.50 m2"] * 3)
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser().parse(path)
            assert result.is_scanned is False
        finally:
            os.unlink(path)

    def test_empty_pdf_detected_as_scanned(self):
        pdf_bytes = _make_empty_pdf(3)
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser().parse(path)
            assert result.is_scanned is True
        finally:
            os.unlink(path)

    def test_force_ocr_overrides_detection(self):
        pdf_bytes = _make_digital_pdf(["Texto digital normal"] * 2)
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser(force_ocr=True).parse(path)
            assert result.is_scanned is True
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Testes de extração digital
# ---------------------------------------------------------------------------

class TestDigitalExtraction:
    def test_extracts_text_blocks(self):
        pdf_bytes = _make_digital_pdf(["DORMITÓRIO 01", "SALA DE JANTAR"])
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser().parse(path)
            all_text = " ".join(b.content for b in result.text_blocks)
            assert "DORMITÓRIO" in all_text or "DORMIT" in all_text
        finally:
            os.unlink(path)

    def test_page_index_recorded(self):
        pdf_bytes = _make_digital_pdf(["Página zero", "Página um"])
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser().parse(path)
            pages = {b.page for b in result.text_blocks}
            assert 0 in pages
        finally:
            os.unlink(path)

    def test_max_pages_limits_extraction(self):
        pdf_bytes = _make_digital_pdf(["Texto A", "Texto B", "Texto C"])
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result_all = PDFParser().parse(path)
            result_one = PDFParser(max_pages=1).parse(path)
            assert len(result_one.text_blocks) <= len(result_all.text_blocks)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Testes de TableData
# ---------------------------------------------------------------------------

class TestTableData:
    def _make_table(self) -> TableData:
        return TableData(
            page=0,
            headers=["AMBIENTE", "ÁREA (m²)", "PISO"],
            rows=[
                ["Sala de Estar", "20.50", "Porcelanato"],
                ["Dormitório 01", "12.30", "Cerâmica"],
            ],
            bbox=None,
        )

    def test_markdown_contains_headers(self):
        md = self._make_table().to_markdown()
        assert "AMBIENTE" in md
        assert "ÁREA" in md

    def test_markdown_contains_rows(self):
        md = self._make_table().to_markdown()
        assert "Sala de Estar" in md
        assert "20.50" in md

    def test_empty_table(self):
        t = TableData(page=0, headers=[], rows=[], bbox=None)
        assert t.to_markdown() == "(tabela vazia)"


# ---------------------------------------------------------------------------
# Testes de PDFParseResult
# ---------------------------------------------------------------------------

class TestPDFParseResult:
    def test_to_text_summary_structure(self):
        pdf_bytes = _make_digital_pdf(["SALA - 15.0 m2"])
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser().parse(path)
            summary = result.to_text_summary()
            assert "=== ARQUIVO PDF ===" in summary
            assert "=== PÁGINA 1 ===" in summary
        finally:
            os.unlink(path)

    def test_total_chars_property(self):
        pdf_bytes = _make_digital_pdf(["ABCDE"])
        path = _save_tmp_pdf(pdf_bytes)
        try:
            result = PDFParser().parse(path)
            assert result.total_chars >= 5
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Testes de erro
# ---------------------------------------------------------------------------

class TestParserErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PDFParser().parse("nao_existe.pdf")

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "test.dxf"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Extensão"):
            PDFParser().parse(f)

    def test_corrupted_pdf(self, tmp_path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a pdf at all %%%")
        with pytest.raises(ValueError, match="abrir"):
            PDFParser().parse(f)


# ---------------------------------------------------------------------------
# Testes de pré-processamento OpenCV (condicional)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _OPENCV_OK, reason="OpenCV não instalado")
class TestPreprocessing:
    def test_preprocess_returns_image(self):
        from PIL import Image as PILImage
        import numpy as np
        img = PILImage.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8), "RGB")
        result = _preprocess_for_ocr(img)
        assert result is not None
