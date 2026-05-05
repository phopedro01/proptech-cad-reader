"""
Batch Processor
===============
Orquestra parsing e análise LLM de múltiplos arquivos DXF/PDF.

As duas fases são separadas intencionalmente:
  1. parse_single()   — rápido, sem custo de API, pode ser executado ao upload
  2. run_llm_batch()  — consome tokens, executado apenas sob demanda

Os FileJob são mutados in-place durante run_llm_batch, o que permite
que o Streamlit session_state reflita o progresso em tempo real.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.parsers.dwg_parser import DXFParser
from src.parsers.pdf_parser import PDFParser
from src.ai.nlp_processor import NLPProcessor

logger = logging.getLogger(__name__)

# Tipo do callback de progresso: (índice_atual, total, nome_arquivo)
ProgressCb = Callable[[int, int, str], None]

_STATUS_ICONS: dict[str, str] = {
    "pending":   "⏳ Aguardando",
    "parsing":   "🔄 Extraindo",
    "ready":     "✅ Pronto",
    "analyzing": "🤖 Analisando",
    "done":      "✅ Concluído",
    "error":     "❌ Erro",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FileJob:
    """Estado de um arquivo individual no pipeline de batch."""

    filename: str
    file_hash: str
    size_bytes: int

    # preenchidos durante o pipeline
    file_type: Optional[str] = None           # "dxf" | "pdf"
    status: str = "pending"
    parse_result: Any = None
    raw_summary: Optional[str] = None
    llm_result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    parse_duration_s: float = 0.0
    llm_duration_s: float = 0.0

    # -----------------------------------------------------------------------
    # Propriedades de conveniência
    # -----------------------------------------------------------------------

    @property
    def is_ready_for_llm(self) -> bool:
        return self.status == "ready" and bool(self.raw_summary)

    @property
    def has_error(self) -> bool:
        return self.status == "error"

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    @property
    def status_label(self) -> str:
        return _STATUS_ICONS.get(self.status, self.status)

    @property
    def size_display(self) -> str:
        kb = self.size_bytes / 1024
        return f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"

    @property
    def type_badge(self) -> str:
        return {"dxf": "🔷 DXF", "pdf": "🔴 PDF"}.get(self.file_type or "", "❓")

    @property
    def total_duration_s(self) -> float:
        return self.parse_duration_s + self.llm_duration_s


@dataclass
class BatchResult:
    """Resultado agregado de uma rodada de análise em lote."""

    jobs: list[FileJob]
    query: str

    # -----------------------------------------------------------------------
    # Filtros
    # -----------------------------------------------------------------------

    @property
    def successful(self) -> list[FileJob]:
        return [j for j in self.jobs if j.is_done]

    @property
    def failed(self) -> list[FileJob]:
        return [j for j in self.jobs if j.has_error]

    @property
    def total(self) -> int:
        return len(self.jobs)

    @property
    def success_rate(self) -> float:
        return len(self.successful) / self.total if self.total else 0.0

    # -----------------------------------------------------------------------
    # Agregação de tabelas
    # -----------------------------------------------------------------------

    def combined_tables(self) -> dict[str, "pd.DataFrame"]:
        """
        Mescla os DataFrames de todos os jobs bem-sucedidos.
        Insere a coluna 'Arquivo' como primeira coluna para rastreabilidade.
        """
        import pandas as pd
        from src.utils.exporters import result_to_dataframe

        buckets: dict[str, list[pd.DataFrame]] = {}
        for job in self.successful:
            if not job.llm_result:
                continue
            stem = Path(job.filename).stem
            frames = result_to_dataframe(job.llm_result)
            for key, df in frames.items():
                copy = df.copy()
                copy.insert(0, "Arquivo", stem)
                buckets.setdefault(key, []).append(copy)

        return {
            key: pd.concat(dfs, ignore_index=True)
            for key, dfs in buckets.items()
        }

    # -----------------------------------------------------------------------
    # Serialização
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "summary": {
                "total": self.total,
                "successful": len(self.successful),
                "failed": len(self.failed),
                "success_rate_pct": round(self.success_rate * 100, 1),
            },
            "results": [
                {
                    "filename": j.filename,
                    "status": j.status,
                    "file_type": j.file_type,
                    "llm_result": j.llm_result,
                    "error": j.error,
                    "timings_s": {
                        "parse": round(j.parse_duration_s, 2),
                        "llm": round(j.llm_duration_s, 2),
                        "total": round(j.total_duration_s, 2),
                    },
                }
                for j in self.jobs
            ],
        }


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """
    Orquestra parsing e análise LLM para múltiplos arquivos.

    Parâmetros de configuração do parser são passados no construtor
    e aplicados a todos os arquivos do batch. A separação entre
    parse_single e run_llm_batch permite que a UI exiba o resumo
    de extração antes de consumir tokens do LLM.
    """

    def __init__(
        self,
        dxf_layers: Optional[list[str]] = None,
        dxf_min_height: float = 0.0,
        pdf_ocr_lang: str = "por",
        pdf_dpi: int = 200,
        pdf_force_ocr: bool = False,
    ) -> None:
        self._dxf_layers = dxf_layers or None
        self._dxf_min_height = dxf_min_height
        self._pdf_ocr_lang = pdf_ocr_lang
        self._pdf_dpi = pdf_dpi
        self._pdf_force_ocr = pdf_force_ocr

    # -----------------------------------------------------------------------
    # Fase 1 — Parse
    # -----------------------------------------------------------------------

    def parse_single(
        self,
        filename: str,
        file_bytes: bytes,
        file_hash: str,
    ) -> FileJob:
        """
        Parseia um único arquivo e retorna um FileJob com status 'ready' ou 'error'.
        O arquivo temporário é criado e removido dentro deste método.
        """
        job = FileJob(
            filename=filename,
            file_hash=file_hash,
            size_bytes=len(file_bytes),
            status="parsing",
        )
        suffix = Path(filename).suffix.lower()
        t0 = time.perf_counter()
        tmp_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(file_bytes)
                tmp_path = f.name

            if suffix == ".dxf":
                result = DXFParser(
                    target_layers=self._dxf_layers,
                    min_text_height=self._dxf_min_height,
                ).parse(tmp_path)
                job.file_type = "dxf"

            elif suffix == ".pdf":
                result = PDFParser(
                    ocr_language=self._pdf_ocr_lang,
                    dpi=self._pdf_dpi,
                    force_ocr=self._pdf_force_ocr,
                ).parse(tmp_path)
                job.file_type = "pdf"

            else:
                raise ValueError(f"Formato não suportado: '{suffix}'")

            job.parse_result = result
            job.raw_summary = result.to_text_summary()
            job.status = "ready"

        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            logger.error("Parse [%s]: %s", filename, exc, exc_info=True)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        job.parse_duration_s = time.perf_counter() - t0
        logger.info(
            "parse_single: %s → %s (%.2fs)",
            filename, job.status, job.parse_duration_s,
        )
        return job

    # -----------------------------------------------------------------------
    # Fase 2 — LLM
    # -----------------------------------------------------------------------

    def run_llm_batch(
        self,
        jobs: list[FileJob],
        query: str,
        on_progress: Optional[ProgressCb] = None,
    ) -> BatchResult:
        """
        Executa o LLM sobre cada job com status='ready'.
        Os jobs são MUTADOS in-place para refletir o progresso.
        Jobs com outros status são passados adiante sem modificação.

        O callback on_progress(current, total, filename) é chamado
        antes de cada arquivo e uma vez ao término (filename='').
        """
        ready = [j for j in jobs if j.is_ready_for_llm]
        total = len(ready)

        if total == 0:
            logger.warning("run_llm_batch: nenhum job pronto para análise.")
            return BatchResult(jobs=jobs, query=query)

        processor = NLPProcessor()

        for idx, job in enumerate(ready):
            if on_progress:
                on_progress(idx, total, job.filename)

            job.status = "analyzing"
            t0 = time.perf_counter()

            try:
                job.llm_result = processor.process(
                    raw_text=job.raw_summary,      # type: ignore[arg-type]
                    user_query=query,
                )
                job.status = "done"
            except Exception as exc:
                job.status = "error"
                job.error = f"LLM: {exc}"
                logger.error("LLM [%s]: %s", job.filename, exc, exc_info=True)
            finally:
                job.llm_duration_s = time.perf_counter() - t0

            logger.info(
                "run_llm_batch: %s → %s (%.2fs)",
                job.filename, job.status, job.llm_duration_s,
            )

        if on_progress:
            on_progress(total, total, "")

        return BatchResult(jobs=jobs, query=query)
