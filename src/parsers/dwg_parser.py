"""
DXF/DWG Parser
==============
Extrai textos, blocos, cotas e polilínhas de arquivos DXF (AutoCAD).

Nota sobre DWG:
  ezdxf suporta apenas DXF (formato aberto). Arquivos .dwg precisam ser
  convertidos antes do processamento. Use uma das opções:
    - AutoCAD / BricsCAD: "Salvar como" → DXF R2010
    - FreeCAD (gratuito): File > Export > DXF
    - ODA File Converter (gratuito): https://www.opendesign.com/guestfiles/oda_file_converter
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import ezdxf
from ezdxf.document import Drawing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex para limpar formatação inline do MTEXT
# Ex: \P (parágrafo), \~ (espaço fixo), {grupos}, \Aln;, \C2; etc.
# ---------------------------------------------------------------------------
_MTEXT_STRIP = re.compile(
    r"\\[A-Za-z]+[^;]*;"   # sequências de controle  \X...;
    r"|[{}]"               # chaves de grupo
    r"|\\[P~]"             # \P e \~
    r"|%%[a-zA-Z0-9]+"     # códigos especiais  %%d %%p %%c
)


# ---------------------------------------------------------------------------
# Data classes — estrutura de saída
# ---------------------------------------------------------------------------

@dataclass
class TextElement:
    content: str
    layer: str
    position: tuple[float, float]
    height: float
    entity_type: str          # "TEXT" | "MTEXT"


@dataclass
class BlockReference:
    name: str
    layer: str
    position: tuple[float, float]
    rotation: float           # graus
    scale: tuple[float, float, float]
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class DimensionElement:
    measurement: float        # valor calculado a partir da geometria
    text_override: str        # texto manual sobreposto pelo desenhista (pode ser "")
    layer: str
    dim_type: str             # Linear | Aligned | Angular | Diameter | Radius …
    start_point: tuple[float, float]
    end_point: tuple[float, float]

    @property
    def display_value(self) -> str:
        return self.text_override if self.text_override else f"{self.measurement:.3f}"


@dataclass
class PolylineElement:
    layer: str
    vertices: list[tuple[float, float]]
    is_closed: bool
    area: float               # 0.0 se aberta
    perimeter: float


@dataclass
class HatchElement:
    layer: str
    pattern_name: str         # ex: "SOLID", "ANSI31", "CROSS"
    boundary_area: float      # área calculada do boundary externo


@dataclass
class DXFParseResult:
    file_path: str
    units: str
    layers: list[str]
    texts: list[TextElement]
    blocks: list[BlockReference]
    dimensions: list[DimensionElement]
    polylines: list[PolylineElement]
    hatches: list[HatchElement]
    errors: list[str]

    # -----------------------------------------------------------------------
    # Serialização
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text_summary(self) -> str:
        """
        Gera um resumo em texto plano adequado para consumo pelo LLM.
        Cada seção é delimitada claramente para facilitar o parsing.
        """
        lines: list[str] = [
            "=== ARQUIVO DXF ===",
            f"Caminho : {self.file_path}",
            f"Unidades: {self.units}",
            f"Layers  : {', '.join(self.layers)}",
        ]

        # Textos
        lines += ["", "=== TEXTOS ==="]
        if self.texts:
            for t in self.texts:
                lines.append(
                    f"[{t.entity_type}] layer={t.layer!r:<20} "
                    f"h={t.height:.2f}  pos=({t.position[0]:.1f},{t.position[1]:.1f})  "
                    f'texto="{t.content}"'
                )
        else:
            lines.append("(nenhum texto encontrado)")

        # Blocos
        lines += ["", "=== BLOCOS / INSERÇÕES ==="]
        if self.blocks:
            for b in self.blocks:
                attrs = "  ".join(f"{k}={v}" for k, v in b.attributes.items())
                lines.append(
                    f"bloco={b.name!r:<20} layer={b.layer!r:<15} "
                    f"rot={b.rotation:.1f}°  pos=({b.position[0]:.1f},{b.position[1]:.1f})"
                    + (f"  attrs: {attrs}" if attrs else "")
                )
        else:
            lines.append("(nenhum bloco encontrado)")

        # Cotas
        lines += ["", "=== COTAS (DIMENSIONS) ==="]
        if self.dimensions:
            for d in self.dimensions:
                lines.append(
                    f"tipo={d.dim_type:<12} layer={d.layer!r:<15} "
                    f"valor={d.display_value}"
                )
        else:
            lines.append("(nenhuma cota encontrada)")

        # Polilínhas
        lines += ["", "=== POLILÍNHAS FECHADAS (possíveis ambientes/áreas) ==="]
        closed = [p for p in self.polylines if p.is_closed]
        if closed:
            for p in closed:
                lines.append(
                    f"layer={p.layer!r:<20} "
                    f"área={p.area:.3f} {self.units}²  "
                    f"perim={p.perimeter:.3f} {self.units}  "
                    f"vértices={len(p.vertices)}"
                )
        else:
            lines.append("(nenhuma polilinha fechada encontrada)")

        # Hachuras
        lines += ["", "=== HACHURAS (padrões de piso/revestimento) ==="]
        if self.hatches:
            for h in self.hatches:
                lines.append(
                    f"layer={h.layer!r:<20} padrão={h.pattern_name!r:<15} "
                    f"área={h.boundary_area:.3f} {self.units}²"
                )
        else:
            lines.append("(nenhuma hachura encontrada)")

        if self.errors:
            lines += ["", "=== AVISOS DE EXTRAÇÃO ==="]
            lines += [f"  ! {e}" for e in self.errors]

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Helpers de análise rápida
    # -----------------------------------------------------------------------

    def closed_polylines_by_layer(self) -> dict[str, list[PolylineElement]]:
        result: dict[str, list[PolylineElement]] = {}
        for p in self.polylines:
            if p.is_closed:
                result.setdefault(p.layer, []).append(p)
        return result

    def blocks_by_name(self) -> dict[str, list[BlockReference]]:
        result: dict[str, list[BlockReference]] = {}
        for b in self.blocks:
            result.setdefault(b.name, []).append(b)
        return result


# ---------------------------------------------------------------------------
# Funções geométricas internas
# ---------------------------------------------------------------------------

def _clean_mtext(raw: str) -> str:
    """Remove formatação inline do MTEXT e normaliza espaços."""
    cleaned = _MTEXT_STRIP.sub(" ", raw)
    return " ".join(cleaned.split())


def _shoelace_area(vertices: list[tuple[float, float]]) -> float:
    """Fórmula de Gauss (shoelace) para área de polígono simples."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = sum(
        vertices[i][0] * vertices[(i + 1) % n][1]
        - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )
    return abs(area) / 2.0


def _polyline_perimeter(
    vertices: list[tuple[float, float]], is_closed: bool
) -> float:
    n = len(vertices)
    if n < 2:
        return 0.0
    total = sum(
        math.hypot(vertices[i + 1][0] - vertices[i][0], vertices[i + 1][1] - vertices[i][1])
        for i in range(n - 1)
    )
    if is_closed:
        total += math.hypot(
            vertices[0][0] - vertices[-1][0],
            vertices[0][1] - vertices[-1][1],
        )
    return total


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class DXFParser:
    """
    Lê um arquivo DXF e extrai elementos estruturais relevantes para
    projetos de engenharia civil: textos, blocos, cotas, polilínhas e hachuras.

    Parâmetros
    ----------
    target_layers : list[str] | None
        Se fornecida, apenas entidades nessas layers são extraídas.
        Aceita wildcards simples (prefixo*) — ex: ["A-*", "COTA*"].
    min_text_height : float
        Altura mínima de texto (unidades do desenho) para inclusão.
        Útil para filtrar textos de dimensionamento muito pequenos.
    closed_polylines_only : bool
        Se True, ignora polilínhas abertas na extração.
    """

    _DIM_TYPES: dict[int, str] = {
        0: "Linear",
        1: "Aligned",
        2: "Angular",
        3: "Diameter",
        4: "Radius",
        5: "Angular3P",
        6: "Ordinate",
    }

    _UNIT_CODES: dict[int, str] = {
        0: "Sem unidade",
        1: "Polegadas",
        2: "Pés",
        4: "Milímetros",
        5: "Centímetros",
        6: "Metros",
        14: "Decímetros",
    }

    def __init__(
        self,
        target_layers: Optional[list[str]] = None,
        min_text_height: float = 0.0,
        closed_polylines_only: bool = False,
    ) -> None:
        self._raw_target = [l.upper() for l in target_layers] if target_layers else None
        self.min_text_height = min_text_height
        self.closed_polylines_only = closed_polylines_only

    # ---- layer filter ------------------------------------------------------

    def _layer_ok(self, layer: str) -> bool:
        if self._raw_target is None:
            return True
        layer_up = layer.upper()
        for pattern in self._raw_target:
            if pattern.endswith("*"):
                if layer_up.startswith(pattern[:-1]):
                    return True
            elif layer_up == pattern:
                return True
        return False

    # ---- public entry point ------------------------------------------------

    def parse(self, file_path: str | Path) -> DXFParseResult:
        """
        Processa o arquivo DXF e retorna um DXFParseResult estruturado.

        Raises
        ------
        FileNotFoundError   : arquivo não existe.
        ValueError          : formato inválido ou arquivo corrompido.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        if path.suffix.lower() == ".dwg":
            raise ValueError(
                "Formato .dwg não é suportado diretamente.\n"
                "Converta para .dxf usando AutoCAD, FreeCAD ou ODA File Converter."
            )
        if path.suffix.lower() != ".dxf":
            raise ValueError(
                f"Extensão não suportada: '{path.suffix}'. Esperado: .dxf"
            )

        try:
            doc: Drawing = ezdxf.readfile(str(path))
        except ezdxf.DXFStructureError as exc:
            raise ValueError(f"Arquivo DXF mal-formado: {exc}") from exc

        units_code = doc.header.get("$INSUNITS", 0)
        units = self._UNIT_CODES.get(units_code, f"Código {units_code}")
        layers = [layer.dxf.name for layer in doc.layers]

        msp = doc.modelspace()
        errors: list[str] = []

        texts = self._extract_texts(msp, errors)
        blocks = self._extract_blocks(msp, errors)
        dimensions = self._extract_dimensions(msp, errors)
        polylines = self._extract_polylines(msp, errors)
        hatches = self._extract_hatches(msp, errors)

        logger.info(
            "DXF parseado: %d textos, %d blocos, %d cotas, %d polilínhas, %d hachuras — %d aviso(s)",
            len(texts), len(blocks), len(dimensions), len(polylines), len(hatches), len(errors),
        )

        return DXFParseResult(
            file_path=str(path),
            units=units,
            layers=layers,
            texts=texts,
            blocks=blocks,
            dimensions=dimensions,
            polylines=polylines,
            hatches=hatches,
            errors=errors,
        )

    # ---- extrators privados ------------------------------------------------

    def _extract_texts(self, msp, errors: list[str]) -> list[TextElement]:
        result: list[TextElement] = []
        for entity in msp:
            layer = entity.dxf.layer
            if not self._layer_ok(layer):
                continue
            try:
                if entity.dxftype() == "TEXT":
                    content = (entity.dxf.text or "").strip()
                    if not content:
                        continue
                    height = entity.dxf.get("height", 0.0)
                    if height < self.min_text_height:
                        continue
                    pos = entity.dxf.insert
                    result.append(
                        TextElement(
                            content=content,
                            layer=layer,
                            position=(float(pos.x), float(pos.y)),
                            height=float(height),
                            entity_type="TEXT",
                        )
                    )

                elif entity.dxftype() == "MTEXT":
                    # plain_mtext() retorna texto sem formatação (ezdxf ≥ 0.17)
                    try:
                        raw = entity.plain_mtext()
                    except AttributeError:
                        raw = entity.text
                    content = _clean_mtext(raw).strip()
                    if not content:
                        continue
                    height = entity.dxf.get("char_height", 0.0)
                    if height < self.min_text_height:
                        continue
                    pos = entity.dxf.insert
                    result.append(
                        TextElement(
                            content=content,
                            layer=layer,
                            position=(float(pos.x), float(pos.y)),
                            height=float(height),
                            entity_type="MTEXT",
                        )
                    )
            except Exception as exc:
                errors.append(f"Texto (layer={layer!r}): {exc}")
                logger.debug("Erro texto", exc_info=True)
        return result

    def _extract_blocks(self, msp, errors: list[str]) -> list[BlockReference]:
        result: list[BlockReference] = []
        for entity in msp:
            if entity.dxftype() != "INSERT":
                continue
            layer = entity.dxf.layer
            if not self._layer_ok(layer):
                continue
            try:
                pos = entity.dxf.insert
                rotation = float(entity.dxf.get("rotation", 0.0))
                sx = float(entity.dxf.get("xscale", 1.0))
                sy = float(entity.dxf.get("yscale", 1.0))
                sz = float(entity.dxf.get("zscale", 1.0))

                attributes: dict[str, str] = {}
                for attrib in entity.attribs:
                    tag = (attrib.dxf.tag or "").strip()
                    value = (attrib.dxf.text or "").strip()
                    if tag:
                        attributes[tag] = value

                result.append(
                    BlockReference(
                        name=entity.dxf.name,
                        layer=layer,
                        position=(float(pos.x), float(pos.y)),
                        rotation=rotation,
                        scale=(sx, sy, sz),
                        attributes=attributes,
                    )
                )
            except Exception as exc:
                errors.append(f"Bloco (layer={layer!r}): {exc}")
                logger.debug("Erro bloco", exc_info=True)
        return result

    def _extract_dimensions(self, msp, errors: list[str]) -> list[DimensionElement]:
        result: list[DimensionElement] = []
        for entity in msp:
            if entity.dxftype() != "DIMENSION":
                continue
            layer = entity.dxf.layer
            if not self._layer_ok(layer):
                continue
            try:
                raw_type = entity.dxf.get("dimtype", 0) & 0x0F
                dim_type = self._DIM_TYPES.get(raw_type, f"Tipo{raw_type}")

                text_override = (entity.dxf.get("text", "") or "").strip()

                # ponto de definição principal
                try:
                    dp = entity.dxf.defpoint
                    start = (float(dp.x), float(dp.y))
                except Exception:
                    start = (0.0, 0.0)

                # segundo ponto de definição (cotas lineares/alinhadas)
                try:
                    dp2 = entity.dxf.defpoint2
                    end = (float(dp2.x), float(dp2.y))
                except Exception:
                    try:
                        dp3 = entity.dxf.defpoint3
                        end = (float(dp3.x), float(dp3.y))
                    except Exception:
                        end = (0.0, 0.0)

                measurement = math.hypot(end[0] - start[0], end[1] - start[1])

                result.append(
                    DimensionElement(
                        measurement=measurement,
                        text_override=text_override,
                        layer=layer,
                        dim_type=dim_type,
                        start_point=start,
                        end_point=end,
                    )
                )
            except Exception as exc:
                errors.append(f"Cota (layer={layer!r}): {exc}")
                logger.debug("Erro cota", exc_info=True)
        return result

    def _extract_polylines(self, msp, errors: list[str]) -> list[PolylineElement]:
        result: list[PolylineElement] = []
        for entity in msp:
            layer = entity.dxf.layer
            if not self._layer_ok(layer):
                continue
            try:
                vertices: Optional[list[tuple[float, float]]] = None
                is_closed = False

                if entity.dxftype() == "LWPOLYLINE":
                    vertices = [(float(p[0]), float(p[1])) for p in entity.get_points()]
                    is_closed = bool(entity.is_closed)

                elif entity.dxftype() == "POLYLINE" and entity.is_2d_polyline:
                    vertices = [
                        (float(v.dxf.location.x), float(v.dxf.location.y))
                        for v in entity.vertices
                    ]
                    is_closed = bool(entity.is_closed)

                if vertices is None or len(vertices) < 2:
                    continue
                if self.closed_polylines_only and not is_closed:
                    continue

                area = _shoelace_area(vertices) if is_closed else 0.0
                perimeter = _polyline_perimeter(vertices, is_closed)

                result.append(
                    PolylineElement(
                        layer=layer,
                        vertices=vertices,
                        is_closed=is_closed,
                        area=area,
                        perimeter=perimeter,
                    )
                )
            except Exception as exc:
                errors.append(f"Polilinha (layer={layer!r}): {exc}")
                logger.debug("Erro polilinha", exc_info=True)
        return result

    def _extract_hatches(self, msp, errors: list[str]) -> list[HatchElement]:
        """Extrai hachuras — frequentemente usadas para indicar tipos de piso/revestimento."""
        result: list[HatchElement] = []
        for entity in msp:
            if entity.dxftype() != "HATCH":
                continue
            layer = entity.dxf.layer
            if not self._layer_ok(layer):
                continue
            try:
                pattern_name = entity.dxf.get("pattern_name", "SOLID") or "SOLID"

                # Calcula área a partir do boundary externo (primeiro path)
                boundary_area = 0.0
                if entity.paths:
                    outer_path = entity.paths[0]
                    # EdgePath (arcos, splines) ou PolylinePath
                    try:
                        pts = list(outer_path.control_points)   # PolylinePath
                    except AttributeError:
                        pts = []
                    if len(pts) >= 3:
                        boundary_area = _shoelace_area([(p[0], p[1]) for p in pts])

                result.append(
                    HatchElement(
                        layer=layer,
                        pattern_name=pattern_name,
                        boundary_area=boundary_area,
                    )
                )
            except Exception as exc:
                errors.append(f"Hachura (layer={layer!r}): {exc}")
                logger.debug("Erro hachura", exc_info=True)
        return result
