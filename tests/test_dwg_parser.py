"""
Testes unitários do DXFParser.
Usa ezdxf para criar documentos DXF sintéticos em memória — não requer arquivo real.
"""

import math
import pytest
import ezdxf

from src.parsers.dwg_parser import DXFParser, DXFParseResult, _shoelace_area, _clean_mtext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dxf_with_text() -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_text("SALA DE ESTAR", dxfattribs={"layer": "A-TEXT", "height": 0.25, "insert": (5, 5, 0)})
    msp.add_text("DORMITÓRIO 01", dxfattribs={"layer": "A-TEXT", "height": 0.25, "insert": (20, 5, 0)})
    return doc


def _make_dxf_with_polyline() -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    # Retângulo 5×4 m = 20 m²
    pts = [(0, 0), (5, 0), (5, 4), (0, 4)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "A-ROOM"})
    return doc


def _make_dxf_with_blocks() -> ezdxf.document.Drawing:
    doc = ezdxf.new("R2010")
    # cria definição do bloco
    blk = doc.blocks.new("PORTA-90")
    blk.add_line((0, 0), (0.9, 0))
    msp = doc.modelspace()
    msp.add_blockref("PORTA-90", (10, 10), dxfattribs={"layer": "A-DOOR", "rotation": 90})
    msp.add_blockref("PORTA-90", (15, 10), dxfattribs={"layer": "A-DOOR", "rotation": 0})
    return doc


# ---------------------------------------------------------------------------
# Helpers de teste (parse em memória sem arquivo)
# ---------------------------------------------------------------------------

def _parse_doc(doc: ezdxf.document.Drawing, **kwargs) -> DXFParseResult:
    """Cria parser e injeta o modelspace diretamente."""
    import tempfile, os
    parser = DXFParser(**kwargs)
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        tmp_path = f.name
    try:
        doc.saveas(tmp_path)
        return parser.parse(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Testes geométricos
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_shoelace_square(self):
        # Quadrado 3×3 → área = 9
        pts = [(0, 0), (3, 0), (3, 3), (0, 3)]
        assert math.isclose(_shoelace_area(pts), 9.0)

    def test_shoelace_triangle(self):
        pts = [(0, 0), (4, 0), (0, 3)]
        assert math.isclose(_shoelace_area(pts), 6.0)

    def test_shoelace_too_few_vertices(self):
        assert _shoelace_area([(0, 0), (1, 1)]) == 0.0

    def test_clean_mtext_strips_codes(self):
        raw = r"SALA\P DE ESTAR{\C2;texto}"
        assert "SALA" in _clean_mtext(raw)
        assert "\\P" not in _clean_mtext(raw)
        assert "{" not in _clean_mtext(raw)


# ---------------------------------------------------------------------------
# Testes de extração
# ---------------------------------------------------------------------------

class TestTextExtraction:
    def test_extracts_text_entities(self):
        doc = _make_dxf_with_text()
        result = _parse_doc(doc)
        contents = [t.content for t in result.texts]
        assert "SALA DE ESTAR" in contents
        assert "DORMITÓRIO 01" in contents

    def test_text_position_preserved(self):
        doc = _make_dxf_with_text()
        result = _parse_doc(doc)
        sala = next(t for t in result.texts if t.content == "SALA DE ESTAR")
        assert math.isclose(sala.position[0], 5.0)
        assert math.isclose(sala.position[1], 5.0)

    def test_layer_filter_excludes_other_layers(self):
        doc = _make_dxf_with_text()
        result = _parse_doc(doc, target_layers=["COTA"])
        assert len(result.texts) == 0


class TestPolylineExtraction:
    def test_closed_polyline_area(self):
        doc = _make_dxf_with_polyline()
        result = _parse_doc(doc)
        closed = [p for p in result.polylines if p.is_closed]
        assert len(closed) == 1
        assert math.isclose(closed[0].area, 20.0, rel_tol=1e-3)

    def test_perimeter(self):
        doc = _make_dxf_with_polyline()
        result = _parse_doc(doc)
        closed = [p for p in result.polylines if p.is_closed]
        # Retângulo 5×4: perímetro = 18
        assert math.isclose(closed[0].perimeter, 18.0, rel_tol=1e-3)


class TestBlockExtraction:
    def test_block_count(self):
        doc = _make_dxf_with_blocks()
        result = _parse_doc(doc)
        porta_blocks = [b for b in result.blocks if b.name == "PORTA-90"]
        assert len(porta_blocks) == 2

    def test_block_rotation(self):
        doc = _make_dxf_with_blocks()
        result = _parse_doc(doc)
        rotations = {b.rotation for b in result.blocks if b.name == "PORTA-90"}
        assert 90.0 in rotations
        assert 0.0 in rotations


# ---------------------------------------------------------------------------
# Testes de erro
# ---------------------------------------------------------------------------

class TestParserErrors:
    def test_file_not_found(self):
        parser = DXFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("nao_existe.dxf")

    def test_dwg_raises_value_error(self, tmp_path):
        dwg_file = tmp_path / "test.dwg"
        dwg_file.write_bytes(b"fake dwg content")
        parser = DXFParser()
        with pytest.raises(ValueError, match=r"(?i)dwg"):
            parser.parse(dwg_file)

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"")
        parser = DXFParser()
        with pytest.raises(ValueError, match="Extensão"):
            parser.parse(f)


# ---------------------------------------------------------------------------
# Teste do resumo em texto
# ---------------------------------------------------------------------------

class TestTextSummary:
    def test_summary_contains_sections(self):
        doc = _make_dxf_with_text()
        result = _parse_doc(doc)
        summary = result.to_text_summary()
        assert "=== TEXTOS ===" in summary
        assert "=== BLOCOS" in summary
        assert "=== COTAS" in summary
        assert "SALA DE ESTAR" in summary
