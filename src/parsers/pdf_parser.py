"""
PDF Parser Module
=================
Extrai textos, tabelas e metadados de arquivos PDF de engenharia civil.

Estratégia de detecção automática:
  - PDF digital   → PyMuPDF (texto nativo, posição, fonte) + pdfplumber (tabelas)
  - PDF escaneado → PyMuPDF renderiza cada página como imagem →
                    OpenCV pré-processamento → pytesseract OCR

Dependências de sistema necessárias para OCR:
  Windows: instalar Tesseract em C:\\Program Files\\Tesseract-OCR\\
  Linux  : sudo apt install tesseract-ocr tesseract-ocr-por
  macOS  : brew install tesseract tesseract-lang
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports opcionais — degradação graciosa se não instalados
# ---------------------------------------------------------------------------

try:
    import pytesseract
    from config.settings import settings as _cfg
    _tess_path = _cfg.TESSERACT_CMD
    if os.path.isfile(_tess_path):
        pytesseract.pytesseract.tesseract_cmd = _tess_path
    _TESSERACT_OK = True
except Exception:
    _TESSERACT_OK = False
    logger.warning("pytesseract não disponível — OCR desabilitado.")

try:
    import cv2
    import numpy as np
    from PIL import Image as _PilImage
    _OPENCV_OK = True
except ImportError:
    _OPENCV_OK = False
    logger.warning("OpenCV/Pillow não disponível — pré-processamento OCR reduzido.")

# Número mínimo de caracteres extraídos por página para considerar PDF digital
_MIN_CHARS_PER_PAGE = 80
# Limite de aviso para arquivos com muitas páginas
_LARGE_FILE_PAGES = 30
# DPI padrão para renderização de imagem ao fazer OCR
_DEFAULT_DPI = 200


# ---------------------------------------------------------------------------
# Data classes de saída
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    content: str
    page: int
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 em pontos PDF
    font_name: str
    font_size: float
    source: str                                # "digital" | "ocr"


@dataclass
class TableData:
    page: int
    headers: list[str]
    rows: list[list[str]]
    bbox: Optional[tuple[float, float, float, float]]

    def to_markdown(self) -> str:
        """Converte para tabela Markdown para consumo do LLM."""
        if not self.headers and not self.rows:
            return "(tabela vazia)"
        col_widths = [
            max(len(str(v)) for v in [h] + [r[i] for r in self.rows if i < len(r)])
            for i, h in enumerate(self.headers)
        ] if self.headers else []

        def _row(cells: list[str]) -> str:
            padded = [str(c).ljust(col_widths[i] if i < len(col_widths) else 8) for i, c in enumerate(cells)]
            return "| " + " | ".join(padded) + " |"

        lines: list[str] = []
        if self.headers:
            lines.append(_row(self.headers))
            lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        for row in self.rows:
            lines.append(_row(row))
        return "\n".join(lines)


@dataclass
class PDFParseResult:
    file_path: str
    total_pages: int
    is_scanned: bool
    text_blocks: list[TextBlock]
    tables: list[TableData]
    errors: list[str]

    # -----------------------------------------------------------------------
    # Serialização
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text_summary(self) -> str:
        """
        Resumo em texto plano organizado por página, para consumo pelo LLM.
        """
        lines: list[str] = [
            "=== ARQUIVO PDF ===",
            f"Caminho      : {self.file_path}",
            f"Total páginas: {self.total_pages}",
            f"Modo         : {'Escaneado (OCR)' if self.is_scanned else 'Digital'}",
        ]

        blocks_by_page: dict[int, list[TextBlock]] = {}
        for b in self.text_blocks:
            blocks_by_page.setdefault(b.page, []).append(b)

        tables_by_page: dict[int, list[TableData]] = {}
        for t in self.tables:
            tables_by_page.setdefault(t.page, []).append(t)

        all_pages = sorted(
            set(blocks_by_page.keys()) | set(tables_by_page.keys())
        )

        for pg in all_pages:
            lines.append(f"\n=== PÁGINA {pg + 1} ===")

            # Textos
            page_texts = blocks_by_page.get(pg, [])
            if page_texts:
                lines.append("--- Textos ---")
                for blk in page_texts:
                    tag = f"[{blk.font_name} {blk.font_size:.1f}pt]" if blk.font_name else ""
                    lines.append(f"{tag} {blk.content}")

            # Tabelas
            page_tables = tables_by_page.get(pg, [])
            if page_tables:
                lines.append("--- Tabelas ---")
                for i, tbl in enumerate(page_tables, 1):
                    lines.append(f"[tabela {i}]")
                    lines.append(tbl.to_markdown())

        if self.errors:
            lines += ["", "=== AVISOS ==="]
            lines += [f"  ! {e}" for e in self.errors]

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Estatísticas rápidas
    # -----------------------------------------------------------------------

    @property
    def total_chars(self) -> int:
        return sum(len(b.content) for b in self.text_blocks)

    @property
    def total_tables(self) -> int:
        return len(self.tables)

    def texts_per_page(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for b in self.text_blocks:
            result[b.page] = result.get(b.page, 0) + 1
        return result


# ---------------------------------------------------------------------------
# Funções auxiliares de pré-processamento (OpenCV)
# ---------------------------------------------------------------------------

def _pil_to_cv2(pil_img) -> "np.ndarray":
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _preprocess_for_ocr(pil_img) -> "np.ndarray":
    """
    Pipeline OpenCV para melhorar qualidade de OCR em plantas escaneadas:
    1. Converte para escala de cinza
    2. Remove ruído (fastNlMeansDenoising)
    3. Binariza com threshold adaptativo (robusto a iluminação desigual)
    """
    if not _OPENCV_OK:
        return pil_img

    bgr = _pil_to_cv2(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Remove ruído antes da binarização
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Threshold adaptativo — lida bem com variações de contraste em plantas
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # Retorna como PIL Image para o pytesseract
    return _PilImage.fromarray(binary)


def _pixmap_to_pil(pix: fitz.Pixmap):
    """Converte PyMuPDF Pixmap para PIL Image."""
    mode = "RGBA" if pix.alpha else "RGB"
    return _PilImage.frombytes(mode, (pix.width, pix.height), pix.samples)


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Lê um arquivo PDF e extrai texto estruturado e tabelas.

    Parâmetros
    ----------
    ocr_language : str
        Idioma(s) para o Tesseract (ex: "por", "eng", "por+eng").
    dpi : int
        Resolução de renderização para OCR. 200–300 é adequado para plantas.
    force_ocr : bool
        Força OCR mesmo em PDFs digitais (útil quando o PDF tem texto corrompido).
    max_pages : int | None
        Limita o número de páginas processadas. None = sem limite.
    """

    def __init__(
        self,
        ocr_language: str = "por",
        dpi: int = _DEFAULT_DPI,
        force_ocr: bool = False,
        max_pages: Optional[int] = None,
    ) -> None:
        self.ocr_language = ocr_language
        self.dpi = dpi
        self.force_ocr = force_ocr
        self.max_pages = max_pages

    # ---- entry point -------------------------------------------------------

    def parse(self, file_path: str | Path) -> PDFParseResult:
        """
        Processa o arquivo PDF e retorna um PDFParseResult estruturado.

        Raises
        ------
        FileNotFoundError : arquivo não existe.
        ValueError        : formato inválido.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Extensão não suportada: '{path.suffix}'. Esperado: .pdf")

        errors: list[str] = []

        try:
            fitz_doc = fitz.open(str(path))
        except Exception as exc:
            raise ValueError(f"Não foi possível abrir o PDF: {exc}") from exc

        total_pages = fitz_doc.page_count
        pages_to_process = (
            min(total_pages, self.max_pages) if self.max_pages else total_pages
        )

        if total_pages > _LARGE_FILE_PAGES:
            errors.append(
                f"Arquivo grande ({total_pages} páginas). "
                f"Processando {pages_to_process} página(s)."
            )

        is_scanned = self.force_ocr or self._detect_scanned(fitz_doc, pages_to_process)

        logger.info(
            "PDF: %d pág. | modo=%s | ocr_disponível=%s",
            total_pages,
            "OCR" if is_scanned else "Digital",
            _TESSERACT_OK,
        )

        if is_scanned:
            if not _TESSERACT_OK:
                errors.append(
                    "PDF escaneado detectado mas pytesseract não está instalado. "
                    "Instale Tesseract OCR e pytesseract para extrair texto."
                )
                text_blocks: list[TextBlock] = []
            else:
                text_blocks = self._extract_ocr(fitz_doc, pages_to_process, errors)
            tables: list[TableData] = []
        else:
            text_blocks = self._extract_digital_text(fitz_doc, pages_to_process, errors)
            tables = self._extract_tables(str(path), pages_to_process, errors)

        fitz_doc.close()

        return PDFParseResult(
            file_path=str(path),
            total_pages=total_pages,
            is_scanned=is_scanned,
            text_blocks=text_blocks,
            tables=tables,
            errors=errors,
        )

    # ---- detecção de modo --------------------------------------------------

    def _detect_scanned(self, doc: fitz.Document, n_pages: int) -> bool:
        """
        Conta caracteres extraídos por página.
        Se a média for < _MIN_CHARS_PER_PAGE, classifica como escaneado.
        Amostra no máximo 5 páginas para rapidez.
        """
        sample = min(n_pages, 5)
        total_chars = 0
        for i in range(sample):
            page = doc[i]
            total_chars += len(page.get_text("text"))
        avg = total_chars / sample if sample else 0
        logger.debug("Média de chars/página: %.1f (limiar=%d)", avg, _MIN_CHARS_PER_PAGE)
        return avg < _MIN_CHARS_PER_PAGE

    # ---- extração digital --------------------------------------------------

    def _extract_digital_text(
        self,
        doc: fitz.Document,
        n_pages: int,
        errors: list[str],
    ) -> list[TextBlock]:
        result: list[TextBlock] = []
        for page_idx in range(n_pages):
            try:
                page = doc[page_idx]
                raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                for block in raw.get("blocks", []):
                    if block.get("type") != 0:   # 0 = texto, 1 = imagem
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = (span.get("text") or "").strip()
                            if not text:
                                continue
                            result.append(
                                TextBlock(
                                    content=text,
                                    page=page_idx,
                                    bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                                    font_name=span.get("font", ""),
                                    font_size=float(span.get("size", 0.0)),
                                    source="digital",
                                )
                            )
            except Exception as exc:
                errors.append(f"Extração digital p.{page_idx + 1}: {exc}")
                logger.debug("Erro extração digital", exc_info=True)
        return result

    # ---- extração OCR ------------------------------------------------------

    def _extract_ocr(
        self,
        doc: fitz.Document,
        n_pages: int,
        errors: list[str],
    ) -> list[TextBlock]:
        result: list[TextBlock] = []
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)

        for page_idx in range(n_pages):
            try:
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                pil_img = _pixmap_to_pil(pix)

                # Pré-processa se OpenCV disponível
                if _OPENCV_OK:
                    processed = _preprocess_for_ocr(pil_img)
                else:
                    processed = pil_img

                # OCR com pytesseract — obtém dados com bounding boxes
                ocr_data = pytesseract.image_to_data(
                    processed,
                    lang=self.ocr_language,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 3",   # segmentação automática de página
                )

                n_items = len(ocr_data["text"])
                for i in range(n_items):
                    conf = int(ocr_data["conf"][i])
                    text = (ocr_data["text"][i] or "").strip()
                    if conf < 30 or not text:
                        continue

                    # Converte coordenadas de pixel para pontos PDF
                    scale = 72.0 / self.dpi
                    x0 = ocr_data["left"][i] * scale
                    y0 = ocr_data["top"][i] * scale
                    w  = ocr_data["width"][i] * scale
                    h  = ocr_data["height"][i] * scale

                    result.append(
                        TextBlock(
                            content=text,
                            page=page_idx,
                            bbox=(x0, y0, x0 + w, y0 + h),
                            font_name="ocr",
                            font_size=float(ocr_data["height"][i] * scale * 0.7),
                            source="ocr",
                        )
                    )
            except Exception as exc:
                errors.append(f"OCR p.{page_idx + 1}: {exc}")
                logger.debug("Erro OCR", exc_info=True)
        return result

    # ---- extração de tabelas -----------------------------------------------

    def _extract_tables(
        self,
        file_path: str,
        n_pages: int,
        errors: list[str],
    ) -> list[TableData]:
        """
        Usa pdfplumber (superior ao PyMuPDF para tabelas nativas).
        Tenta estratégia "lines" primeiro; cai para "text" se não encontrar nada.
        """
        result: list[TableData] = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_idx in range(min(n_pages, len(pdf.pages))):
                    plumber_page = pdf.pages[page_idx]

                    tables = self._try_extract_tables(plumber_page, errors, page_idx)
                    for raw_table in tables:
                        if not raw_table or not any(raw_table):
                            continue

                        # Normaliza células: remove None e quebras de linha
                        normalized = [
                            [
                                str(cell or "").replace("\n", " ").strip()
                                for cell in row
                            ]
                            for row in raw_table
                            if row and any(cell for cell in row)
                        ]

                        if len(normalized) < 2:
                            continue

                        # Primeira linha como cabeçalho se não for só números
                        first_row = normalized[0]
                        if any(cell and not cell.replace(".", "").replace(",", "").isnumeric()
                               for cell in first_row):
                            headers = first_row
                            rows = normalized[1:]
                        else:
                            headers = [f"Col{i+1}" for i in range(len(first_row))]
                            rows = normalized

                        result.append(
                            TableData(
                                page=page_idx,
                                headers=headers,
                                rows=rows,
                                bbox=None,
                            )
                        )
        except Exception as exc:
            errors.append(f"Extração de tabelas (pdfplumber): {exc}")
            logger.debug("Erro tabelas pdfplumber", exc_info=True)

        return result

    def _try_extract_tables(self, page, errors: list[str], page_idx: int) -> list:
        """Tenta múltiplas estratégias de detecção de tabelas."""
        strategies = [
            {"vertical_strategy": "lines",   "horizontal_strategy": "lines"},
            {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
            {"vertical_strategy": "text",    "horizontal_strategy": "text"},
        ]
        for strategy in strategies:
            try:
                tables = page.extract_tables(strategy)
                if tables:
                    return tables
            except Exception as exc:
                errors.append(f"Tabela p.{page_idx + 1} (estratégia {strategy}): {exc}")
        return []
