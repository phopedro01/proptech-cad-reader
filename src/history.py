"""
Histórico de consultas por sessão.

HistoryEntry armazena uma consulta completa com seu BatchResult,
permitindo que o usuário reveja resultados anteriores sem re-executar o LLM.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.batch.batch_processor import BatchResult

_MAX_HISTORY = 20   # limite de entradas por sessão


@dataclass
class HistoryEntry:
    query: str
    files: list[str]           # nomes dos arquivos analisados
    successful: int
    failed: int
    batch_result: "BatchResult"
    template: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # -----------------------------------------------------------------------
    # Propriedades de exibição
    # -----------------------------------------------------------------------

    @property
    def time_label(self) -> str:
        return self.timestamp.strftime("%H:%M")

    @property
    def date_label(self) -> str:
        return self.timestamp.strftime("%d/%m %H:%M")

    @property
    def query_preview(self) -> str:
        """Versão truncada da consulta para exibição compacta."""
        return self.query[:55] + ("…" if len(self.query) > 55 else "")

    @property
    def files_preview(self) -> str:
        """Lista de arquivos truncada para caber na sidebar."""
        shown = self.files[:3]
        extra = len(self.files) - 3
        label = ", ".join(shown)
        return label + (f" +{extra}" if extra > 0 else "")

    @property
    def status_summary(self) -> str:
        total = self.successful + self.failed
        if self.failed == 0:
            return f"{total} arquivo(s) · ✅ todos ok"
        return f"{total} arquivo(s) · {self.successful} ok · ❌ {self.failed} erro(s)"


def append_entry(history: list[HistoryEntry], entry: HistoryEntry) -> None:
    """
    Adiciona uma entrada ao histórico respeitando _MAX_HISTORY.
    Remove a entrada mais antiga se o limite for atingido.
    Descarta entradas duplicadas (mesma query + mesmos arquivos).
    """
    # Evita duplicata imediata (duplo clique no botão)
    if history and history[-1].query == entry.query and history[-1].files == entry.files:
        history[-1] = entry   # substitui pelo mais recente
        return

    history.append(entry)

    if len(history) > _MAX_HISTORY:
        history.pop(0)
