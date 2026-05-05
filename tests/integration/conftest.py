"""
Fixtures compartilhadas para testes de integração.

Todos os arquivos sintéticos são gerados em memória uma única vez
por sessão (scope="session") para manter os testes rápidos.

Nomenclatura:
  dxf_bytes_*  — arquivos DXF com conteúdo específico
  pdf_bytes_*  — arquivos PDF digitais
  llm_resp_*   — respostas JSON realistas simulando o LLM
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import ezdxf
import fitz
import pytest


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _dxf_to_bytes(doc: ezdxf.document.Drawing) -> bytes:
    """Salva Drawing em arquivo temporário e retorna os bytes."""
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
        path = f.name
    try:
        doc.saveas(path)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Fixtures de arquivos DXF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dxf_sala() -> bytes:
    """
    DXF de uma sala de estar.
    Conteúdo garantido: texto 'SALA DE ESTAR', polilinha fechada 5×4 m²,
    bloco de porta, texto de piso.
    """
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6   # Metros
    msp = doc.modelspace()

    msp.add_text("SALA DE ESTAR", dxfattribs={
        "layer": "A-TEXT", "height": 0.25, "insert": (1, 1, 0),
    })
    msp.add_text("AREA=20.00M2", dxfattribs={
        "layer": "A-TEXT", "height": 0.15, "insert": (1, 0.6, 0),
    })
    msp.add_text("Piso: Porcelanato 60x60", dxfattribs={
        "layer": "A-REVESTIMENTO", "height": 0.15, "insert": (1, 0.2, 0),
    })

    # Polilinha 5×4 = 20 m²
    msp.add_lwpolyline(
        [(0, 0), (5, 0), (5, 4), (0, 4)], close=True,
        dxfattribs={"layer": "A-ROOM"},
    )

    # Porta
    blk = doc.blocks.new("P-90-SALA")
    blk.add_line((0, 0), (0.9, 0))
    msp.add_blockref("P-90-SALA", (0, 2), dxfattribs={
        "layer": "A-DOOR", "rotation": 0,
    })

    return _dxf_to_bytes(doc)


@pytest.fixture(scope="session")
def dxf_quarto() -> bytes:
    """
    DXF de um dormitório.
    Conteúdo garantido: texto 'DORMITÓRIO 01', polilinha 3×4 m² = 12 m².
    """
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()

    msp.add_text("DORMITÓRIO 01", dxfattribs={
        "layer": "A-TEXT", "height": 0.25, "insert": (1, 1, 0),
    })
    msp.add_text("AREA=12.00M2", dxfattribs={
        "layer": "A-TEXT", "height": 0.15, "insert": (1, 0.6, 0),
    })
    msp.add_text("Piso: Cerâmica 45x45", dxfattribs={
        "layer": "A-REVESTIMENTO", "height": 0.15, "insert": (1, 0.2, 0),
    })

    # Polilinha 3×4 = 12 m²
    msp.add_lwpolyline(
        [(0, 0), (3, 0), (3, 4), (0, 4)], close=True,
        dxfattribs={"layer": "A-ROOM"},
    )

    blk = doc.blocks.new("P-90-QUARTO")
    blk.add_line((0, 0), (0.9, 0))
    msp.add_blockref("P-90-QUARTO", (2, 0), dxfattribs={
        "layer": "A-DOOR", "rotation": 90,
    })

    return _dxf_to_bytes(doc)


@pytest.fixture(scope="session")
def dxf_multiambiente() -> bytes:
    """
    DXF com sala + quarto na mesma planta, layers distintas, 3 portas, 2 janelas.
    Usado para testes de extração de esquadrias.
    """
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()

    # Textos
    for text, layer, pos in [
        ("SALA DE ESTAR",   "A-TEXT", (1, 8, 0)),
        ("DORMITÓRIO 01",   "A-TEXT", (1, 3, 0)),
        ("COZINHA",         "A-TEXT", (7, 8, 0)),
        ("Piso: Porcelanato", "A-REVESTIMENTO", (1, 7.5, 0)),
        ("Piso: Cerâmica",    "A-REVESTIMENTO", (1, 2.5, 0)),
    ]:
        msp.add_text(text, dxfattribs={"layer": layer, "height": 0.25, "insert": pos})

    # Polilínhas
    for pts, layer in [
        ([(0, 5), (5, 5), (5, 10), (0, 10)], "A-ROOM"),   # Sala  5×5=25m²
        ([(0, 0), (4, 0), (4, 5), (0, 5)],  "A-ROOM"),   # Quarto 4×5=20m²
        ([(5, 5), (9, 5), (9, 10), (5, 10)], "A-ROOM"),   # Cozinha 4×5=20m²
    ]:
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})

    # Blocos — 3 portas + 2 janelas
    blk_p = doc.blocks.new("PORTA")
    blk_p.add_line((0, 0), (0.9, 0))
    blk_j = doc.blocks.new("JANELA")
    blk_j.add_line((0, 0), (1.2, 0))

    for pos, rot, block in [
        ((0, 7), 0,  "PORTA"),
        ((0, 2), 0,  "PORTA"),
        ((5, 7), 90, "PORTA"),
        ((2, 0), 0,  "JANELA"),
        ((7, 5), 90, "JANELA"),
    ]:
        msp.add_blockref(block, pos, dxfattribs={"layer": "A-DOOR", "rotation": rot})

    return _dxf_to_bytes(doc)


# ---------------------------------------------------------------------------
# Fixtures de arquivos PDF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_memorial() -> bytes:
    """
    PDF digital (2 páginas) com memorial descritivo de projeto.
    Página 1: ambientes e áreas. Página 2: esquadrias.
    """
    doc = fitz.open()

    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 80),  "MEMORIAL DESCRITIVO — RESIDÊNCIA UNIFAMILIAR", fontsize=14)
    p1.insert_text((50, 120), "AMBIENTES E REVESTIMENTOS", fontsize=12)
    p1.insert_text((50, 150), "Sala de Estar      20.50 m²   Porcelanato 60x60cm", fontsize=10)
    p1.insert_text((50, 170), "Dormitório 01      12.30 m²   Cerâmica 45x45cm",    fontsize=10)
    p1.insert_text((50, 190), "Dormitório 02      10.80 m²   Cerâmica 45x45cm",    fontsize=10)
    p1.insert_text((50, 210), "Cozinha             9.20 m²   Porcelanato 60x60cm", fontsize=10)
    p1.insert_text((50, 230), "Banheiro Social     4.50 m²   Cerâmica 20x20cm",    fontsize=10)
    p1.insert_text((50, 260), "ÁREA TOTAL: 57.30 m²", fontsize=11)

    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 80),  "ESQUADRIAS", fontsize=12)
    p2.insert_text((50, 120), "PORTAS",  fontsize=11)
    p2.insert_text((50, 145), "P01  Madeira          0.90 x 2.10 m   3 unidades", fontsize=10)
    p2.insert_text((50, 165), "P02  Madeira c/ vidro 0.80 x 2.10 m   1 unidade",  fontsize=10)
    p2.insert_text((50, 195), "JANELAS", fontsize=11)
    p2.insert_text((50, 220), "J01  Alumínio         1.20 x 1.20 m   2 unidades", fontsize=10)
    p2.insert_text((50, 240), "J02  Alumínio         1.50 x 1.20 m   1 unidade",  fontsize=10)

    return doc.tobytes()


@pytest.fixture(scope="session")
def pdf_vazio() -> bytes:
    """PDF sem texto (simula PDF escaneado sem OCR disponível)."""
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Fixtures de respostas LLM (JSON realista)
# ---------------------------------------------------------------------------

@pytest.fixture
def llm_resp_ambientes() -> dict[str, Any]:
    return {
        "ambientes": [
            {"nome": "Sala de Estar", "area_m2": 20.5, "tipo_piso": "Porcelanato 60x60"},
            {"nome": "Dormitório 01", "area_m2": 12.3, "tipo_piso": "Cerâmica 45x45"},
        ],
        "area_total_m2": 32.8,
        "unidade_original": "Metros",
        "confianca": "alta",
        "avisos": [],
        "_meta": {
            "template": "ambientes_e_areas",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        },
    }


@pytest.fixture
def llm_resp_esquadrias() -> dict[str, Any]:
    return {
        "esquadrias": [
            {"tipo": "PORTA",  "quantidade": 3, "dimensao": "0.90x2.10m", "material": "MADEIRA",  "ambiente": None, "observacoes": None},
            {"tipo": "JANELA", "quantidade": 2, "dimensao": "1.20x1.20m", "material": "ALUMÍNIO", "ambiente": None, "observacoes": None},
        ],
        "total_portas": 3,
        "total_janelas": 2,
        "confianca": "alta",
        "avisos": [],
        "_meta": {
            "template": "quantitativo_esquadrias",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        },
    }
