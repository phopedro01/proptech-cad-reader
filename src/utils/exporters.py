"""Utilitários de exportação — Excel e JSON (single-file e batch)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def result_to_dataframe(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Converte o dict retornado pelo NLPProcessor em DataFrames prontos para exibição.
    Retorna um dict {nome_da_tabela: DataFrame}.
    """
    frames: dict[str, pd.DataFrame] = {}

    for key, value in result.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            frames[key] = pd.DataFrame(value)

    return frames


def export_to_excel(
    result: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Exporta todas as listas do resultado para abas de um arquivo Excel."""
    output_path = Path(output_path)
    frames = result_to_dataframe(result)

    if not frames:
        raise ValueError("Nenhuma tabela encontrada no resultado para exportar.")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in frames.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return output_path


def export_to_json(
    result: dict[str, Any],
    output_path: str | Path,
    indent: int = 2,
) -> Path:
    """Exporta o resultado completo como JSON formatado."""
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# Exportação em lote
# ---------------------------------------------------------------------------

def _safe_sheet_name(name: str, used: set[str], max_len: int = 31) -> str:
    """
    Gera um nome de aba Excel único e válido (máx 31 chars, sem /\\?*[]').
    Adiciona sufixo numérico se necessário para garantir unicidade.
    """
    clean = re.sub(r"[/\\?*\[\]']", "_", name)[:max_len]
    candidate = clean
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = clean[: max_len - len(suffix)] + suffix
        counter += 1
    used.add(candidate)
    return candidate


def export_batch_to_excel(
    combined_tables: "dict[str, pd.DataFrame]",
    per_file: "list[tuple[str, dict[str, Any]]]",
    output_path: "str | Path",
    summary_rows: "list[dict[str, Any]] | None" = None,
    avisos_rows: "list[dict[str, Any]] | None" = None,
) -> Path:
    """
    Cria um Excel com as seguintes abas (na ordem):

    1. RESUMO          — status de todos os arquivos (parse, LLM, tempo, erro)
    2. COMB_{chave}    — tabelas mescladas de todos os arquivos (coluna 'Arquivo')
    3. {stem}_{chave}  — tabela individual por arquivo
    4. AVISOS          — avisos e alertas emitidos pelo LLM por arquivo

    Parâmetros
    ----------
    combined_tables : resultado de BatchResult.combined_tables()
    per_file        : lista de (filename, llm_result) dos jobs bem-sucedidos
    output_path     : caminho do .xlsx a ser gerado
    summary_rows    : linhas para a aba RESUMO
    avisos_rows     : linhas para a aba AVISOS
    """
    output_path = Path(output_path)
    used_names: set[str] = set()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 1. Aba RESUMO (sempre presente)
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(
                writer, sheet_name="RESUMO", index=False
            )
            used_names.add("RESUMO")

        # 2. Abas combinadas
        for key, df in combined_tables.items():
            sheet = _safe_sheet_name(f"COMB_{key}", used_names)
            df.to_excel(writer, sheet_name=sheet, index=False)

        # 3. Abas por arquivo
        for filename, llm_result in per_file:
            stem = Path(filename).stem
            frames = result_to_dataframe(llm_result)
            for key, df in frames.items():
                sheet = _safe_sheet_name(f"{stem}_{key}", used_names)
                df.to_excel(writer, sheet_name=sheet, index=False)

        # 4. Aba AVISOS (apenas se houver)
        if avisos_rows:
            sheet = _safe_sheet_name("AVISOS", used_names)
            pd.DataFrame(avisos_rows).to_excel(
                writer, sheet_name=sheet, index=False
            )

    return output_path
