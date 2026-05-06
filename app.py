"""
PropTech — Leitor Inteligente de Projetos de Engenharia Civil
=============================================================
Interface Streamlit com suporte a análise em lote de múltiplos DXF/PDF.

Fluxo:
  1. Upload de um ou mais arquivos (DXF / PDF)
  2. Parse automático ao upload — reutiliza cache por hash MD5
  3. Consulta em linguagem natural → LLM analisa todos os arquivos
  4. Resultados: visão combinada (com coluna "Arquivo") + por arquivo
  5. Download: Excel multi-aba ou JSON completo
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página — deve ser a primeira chamada Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PropTech — Leitor de Projetos",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports do projeto
# ---------------------------------------------------------------------------

from config.settings import settings
from src.batch.batch_processor import BatchProcessor, BatchResult, FileJob
from src.ai.prompts import TEMPLATE_REGISTRY, select_prompt_template
from src.history import HistoryEntry, append_entry
from src.utils.exporters import export_batch_to_excel, result_to_dataframe

# ---------------------------------------------------------------------------
# CSS mínimo
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .confidence-high   { color: #15803d; font-weight: 700; }
    .confidence-medium { color: #b45309; font-weight: 700; }
    .confidence-low    { color: #b91c1c; font-weight: 700; }
    .divider           { border-top: 2px solid #e2e8f0; margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Consultas pré-definidas
# ---------------------------------------------------------------------------

PRESET_QUERIES: dict[str, str] = {
    "Ambientes e Áreas":       "Extraia uma tabela com todos os ambientes, suas áreas em m² e o tipo de piso.",
    "Portas e Janelas":        "Liste todas as portas e janelas com dimensões, materiais e ambiente.",
    "Revestimentos":           "Identifique todos os tipos de revestimento (piso, parede, teto) com área estimada.",
    "Resumo Geral":            "Resuma o projeto: tipo de edificação, pavimentos, área total, principais ambientes.",
    "Consulta Livre":          "",
}

# ---------------------------------------------------------------------------
# Inicialização do Session State
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        # lista de FileJob — source of truth do batch
        "jobs": [],
        # frozenset de hashes ativos, para detectar mudanças de seleção
        "prev_hashes": frozenset(),
        # resultado do último run_llm_batch
        "batch_result": None,
        # última consulta digitada
        "last_query": "",
        # list[HistoryEntry] — consultas desta sessão
        "query_history": [],
        # True quando batch_result foi restaurado do histórico (não é novo)
        "is_restored_result": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _batch_excel_bytes(batch_result: BatchResult) -> bytes | None:
    """Gera Excel em memória com RESUMO, dados combinados, por arquivo e AVISOS."""
    try:
        combined = batch_result.combined_tables()
        per_file = [
            (j.filename, j.llm_result)
            for j in batch_result.successful
            if j.llm_result
        ]

        # Aba RESUMO — um linha por arquivo
        summary_rows = [
            {
                "Arquivo":   j.filename,
                "Tipo":      (j.file_type or "").upper(),
                "Status":    "OK" if j.is_done else "Erro",
                "Confiança": (j.llm_result or {}).get("confianca", "") if j.llm_result else "",
                "Parse (s)": round(j.parse_duration_s, 2),
                "LLM (s)":   round(j.llm_duration_s, 2),
                "Total (s)": round(j.total_duration_s, 2),
                "Erro":      j.error or "",
            }
            for j in batch_result.jobs
        ]

        # Aba AVISOS — um aviso por linha
        avisos_rows = [
            {"Arquivo": j.filename, "Aviso": aviso}
            for j in batch_result.jobs
            if j.llm_result
            for aviso in (j.llm_result.get("avisos") or [])
        ]

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            tmp = f.name
        export_batch_to_excel(combined, per_file, tmp,
                              summary_rows=summary_rows,
                              avisos_rows=avisos_rows or None)
        with open(tmp, "rb") as f:
            data = f.read()
        os.unlink(tmp)
        return data
    except Exception:
        return None


def _confidence_html(level: str) -> str:
    css = {
        "alta": "confidence-high", "high": "confidence-high",
        "media": "confidence-medium", "medium": "confidence-medium",
        "baixa": "confidence-low", "low": "confidence-low",
    }.get((level or "").lower(), "confidence-medium")
    labels = {
        "alta": "Alta", "high": "Alta",
        "media": "Média", "medium": "Média",
        "baixa": "Baixa", "low": "Baixa",
    }
    label = labels.get((level or "").lower(), level or "—")
    return f'<span class="{css}">● {label}</span>'


def _restore_history_entry(entry: HistoryEntry) -> None:
    """Carrega resultado histórico no estado atual e aciona re-render."""
    st.session_state.batch_result = entry.batch_result
    st.session_state.last_query = entry.query
    st.session_state.is_restored_result = True
    st.rerun()


def _make_processor(cfg: dict[str, Any]) -> BatchProcessor:
    return BatchProcessor(
        dxf_layers=cfg["dxf_layers"] or None,
        dxf_min_height=cfg["dxf_min_height"],
        pdf_ocr_lang=cfg["pdf_ocr_lang"],
        pdf_dpi=cfg["pdf_dpi"],
        pdf_force_ocr=cfg["pdf_force_ocr"],
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.title("⚙️ Configurações")

        # LLM
        st.subheader("Modelo de IA")
        provider = st.selectbox(
            "Provedor",
            ["openai", "gemini"],
            index=0 if settings.LLM_PROVIDER == "openai" else 1,
        )

        if provider == "openai":
            model = st.selectbox("Modelo", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"])
            api_key = st.text_input(
                "OpenAI API Key", value=settings.OPENAI_API_KEY,
                type="password", help="platform.openai.com",
            )
            settings.LLM_PROVIDER = "openai"
            settings.OPENAI_MODEL = model
            settings.OPENAI_API_KEY = api_key
        else:
            model = st.selectbox("Modelo", ["gemini-1.5-pro", "gemini-1.5-flash"])
            api_key = st.text_input(
                "Google API Key", value=settings.GOOGLE_API_KEY,
                type="password", help="aistudio.google.com",
            )
            settings.LLM_PROVIDER = "gemini"
            settings.GEMINI_MODEL = model
            settings.GOOGLE_API_KEY = api_key

        api_configured = bool(api_key.strip())
        if not api_configured:
            st.warning("Informe a API Key para habilitar análise com IA.")

        st.divider()

        # DXF
        st.subheader("Filtros DXF")
        layers_raw = st.text_input(
            "Layers alvo",
            placeholder="A-TEXT, A-ROOM, COTA* (vazio = todas)",
            help="Wildcard com *. Aplica-se a novos uploads.",
        )
        min_height = st.number_input(
            "Altura mínima de texto", min_value=0.0, max_value=100.0,
            value=0.0, step=0.5,
        )

        st.divider()

        # PDF / OCR
        st.subheader("PDF / OCR")
        ocr_lang = st.selectbox("Idioma OCR", ["por", "eng", "por+eng"])
        pdf_dpi = st.slider("Resolução (DPI)", 100, 400, 200, 50)
        force_ocr = st.checkbox("Forçar OCR em PDF digital")

        st.divider()
        st.caption("PropTech CAD/PDF Reader v1.0")
        st.caption("ezdxf · PyMuPDF · LangChain")

        # ── Histórico de consultas ──────────────────────────────────────────
        st.divider()
        history: list[HistoryEntry] = st.session_state.query_history

        col_h, col_clr = st.columns([5, 1])
        col_h.markdown(f"**📜 Histórico** ({len(history)})")
        if history and col_clr.button(
            "🗑", key="clr_hist", help="Limpar todo o histórico"
        ):
            st.session_state.query_history.clear()
            # Se estava exibindo resultado do histórico, limpa também
            if st.session_state.is_restored_result:
                st.session_state.batch_result = None
                st.session_state.is_restored_result = False
            st.rerun()

        if not history:
            st.caption("Nenhuma consulta realizada nesta sessão.")
        else:
            # Mais recente no topo — sem re-indexar a lista original
            for entry in reversed(history):
                label = f"🕐 {entry.time_label} · {entry.query_preview}"
                with st.expander(label, expanded=False):
                    st.caption(entry.status_summary)
                    st.caption(entry.files_preview)

                    b_col, d_col = st.columns(2)
                    if b_col.button(
                        "▶ Rever", key=f"rv_{entry.id}",
                        use_container_width=True,
                        help="Carregar este resultado na área principal",
                    ):
                        _restore_history_entry(entry)

                    if d_col.button(
                        "🗑", key=f"dl_{entry.id}",
                        use_container_width=True,
                        help="Remover esta entrada",
                    ):
                        st.session_state.query_history.remove(entry)
                        # Se o resultado atual vem desta entrada, limpa
                        if (
                            st.session_state.is_restored_result
                            and st.session_state.batch_result is entry.batch_result
                        ):
                            st.session_state.batch_result = None
                            st.session_state.is_restored_result = False
                        st.rerun()

    return {
        "api_configured": api_configured,
        "dxf_layers": [l.strip() for l in layers_raw.split(",") if l.strip()],
        "dxf_min_height": min_height,
        "pdf_ocr_lang": ocr_lang,
        "pdf_dpi": pdf_dpi,
        "pdf_force_ocr": force_ocr,
    }


# ---------------------------------------------------------------------------
# Upload e parse automático
# ---------------------------------------------------------------------------

def render_upload(cfg: dict[str, Any]) -> None:
    st.header("📁 Upload de Projetos")
    st.caption(
        "Selecione um ou mais arquivos DXF / PDF. "
        "Arquivos já carregados não são re-parseados ao adicionar novos."
    )

    uploaded_files = st.file_uploader(
        "Arraste os arquivos ou clique para selecionar",
        type=["dxf", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.info("Aguardando upload de arquivos…")
        if st.session_state.jobs:
            st.session_state.jobs = []
            st.session_state.prev_hashes = frozenset()
            st.session_state.batch_result = None
        return

    # Lê todos os bytes enquanto os handles estão válidos
    files_data: list[tuple[str, bytes]] = [
        (f.name, f.read()) for f in uploaded_files
    ]
    current_hashes = frozenset(_hash_bytes(d) for _, d in files_data)

    # Detecta mudança na seleção de arquivos
    if current_hashes != st.session_state.prev_hashes:
        st.session_state.batch_result = None
        st.session_state.prev_hashes = current_hashes

    # Mapeia hash → job já parseado
    existing: dict[str, FileJob] = {
        j.file_hash: j for j in st.session_state.jobs
    }

    # Identifica quais arquivos ainda precisam ser parseados
    pending: list[tuple[str, bytes, str]] = []
    for name, data in files_data:
        h = _hash_bytes(data)
        if h not in existing:
            pending.append((name, data, h))

    # Parse incremental — somente arquivos novos
    if pending:
        processor = _make_processor(cfg)
        progress_bar = st.progress(0.0, text="Iniciando extração…")
        status_text = st.empty()

        for i, (name, data, h) in enumerate(pending):
            status_text.markdown(
                f"📂 Extraindo **{name}** ({i + 1}/{len(pending)})…"
            )
            job = processor.parse_single(name, data, h)
            existing[h] = job
            progress_bar.progress(
                (i + 1) / len(pending),
                text=f"Extraído: {name}",
            )

        progress_bar.empty()
        status_text.empty()

    # Reconstrói a lista preservando a ordem de upload
    st.session_state.jobs = [
        existing[_hash_bytes(data)]
        for _, data in files_data
        if _hash_bytes(data) in existing
    ]


# ---------------------------------------------------------------------------
# Lista de arquivos
# ---------------------------------------------------------------------------

def render_file_list() -> None:
    jobs: list[FileJob] = st.session_state.jobs
    if not jobs:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader(f"📋 Arquivos ({len(jobs)})")

    # Tabela de status
    rows = [
        {
            "#": i + 1,
            "Arquivo": j.filename,
            "Tamanho": j.size_display,
            "Tipo": j.type_badge,
            "Status": j.status_label,
            "Parse (s)": f"{j.parse_duration_s:.1f}" if j.parse_duration_s else "—",
        }
        for i, j in enumerate(jobs)
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Parse (s)": st.column_config.TextColumn(width="small"),
        },
    )

    # Erros de parse expandíveis
    parse_errors = [(j.filename, j.error) for j in jobs if j.has_error and j.llm_result is None]
    if parse_errors:
        with st.expander(f"❌ {len(parse_errors)} erro(s) de extração"):
            for fname, err in parse_errors:
                st.error(f"**{fname}**: {err}")


# ---------------------------------------------------------------------------
# Estatísticas agregadas da extração
# ---------------------------------------------------------------------------

def render_aggregate_stats() -> None:
    jobs: list[FileJob] = st.session_state.jobs
    ready = [j for j in jobs if j.is_ready_for_llm]
    errors = [j for j in jobs if j.has_error]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de arquivos", len(jobs))
    col2.metric("Prontos para análise", len(ready))
    col3.metric("DXF", sum(1 for j in jobs if j.file_type == "dxf"))
    col4.metric("PDF", sum(1 for j in jobs if j.file_type == "pdf"))

    if errors:
        st.warning(
            f"{len(errors)} arquivo(s) com erro de extração não serão incluídos na análise.",
            icon="⚠️",
        )

    # Expanders com resumo de cada arquivo
    with st.expander("Ver resumo da extração por arquivo"):
        for job in jobs:
            if not job.is_ready_for_llm:
                continue
            with st.container():
                st.markdown(f"**{job.type_badge} {job.filename}**")
                if job.raw_summary:
                    st.code(job.raw_summary[:1500] + ("…" if len(job.raw_summary) > 1500 else ""),
                            language="text")
                st.divider()


# ---------------------------------------------------------------------------
# Seção de consulta
# ---------------------------------------------------------------------------

def render_query_section(cfg: dict[str, Any]) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("💬 Consulta ao Lote")

    # Presets
    st.caption("Consultas rápidas:")
    cols = st.columns(len(PRESET_QUERIES))
    for col, (label, text) in zip(cols, PRESET_QUERIES.items()):
        if col.button(label, use_container_width=True, key=f"preset_{label}"):
            st.session_state.last_query = text

    user_query = st.text_area(
        "Consulta",
        value=st.session_state.last_query,
        height=80,
        placeholder="Ex: Extraia uma tabela com todos os ambientes e suas áreas",
        label_visibility="collapsed",
    )
    st.session_state.last_query = user_query

    # Contagem de arquivos prontos
    ready_count = sum(1 for j in st.session_state.jobs if j.is_ready_for_llm)

    # Validação e dica
    reasons: list[str] = []
    if not cfg["api_configured"]:
        reasons.append("configure a API Key na barra lateral")
    if not user_query.strip():
        reasons.append("escreva ou selecione uma consulta")
    if ready_count == 0:
        reasons.append("nenhum arquivo pronto para análise")

    can_run = len(reasons) == 0

    if not can_run:
        st.caption(f"⚠️ Para analisar: {' · '.join(reasons)}.")

    label = f"▶ Analisar {ready_count} arquivo(s) em Lote"
    if st.button(label, type="primary", disabled=not can_run):
        _run_batch_llm(user_query, cfg)


def _run_batch_llm(query: str, cfg: dict[str, Any]) -> None:
    """Executa o LLM em todos os jobs prontos, com progress bar em tempo real."""
    jobs: list[FileJob] = st.session_state.jobs
    ready = [j for j in jobs if j.is_ready_for_llm]
    total = len(ready)

    progress_bar = st.progress(0.0, text="Iniciando análise com IA…")
    status_text = st.empty()

    def on_progress(current: int, total_: int, filename: str) -> None:
        if filename:
            frac = current / total_ if total_ else 0
            progress_bar.progress(frac, text=f"🤖 Analisando: {filename} ({current + 1}/{total_})")
            status_text.empty()
        else:
            progress_bar.progress(1.0, text="✅ Análise concluída!")

    try:
        processor = _make_processor(cfg)
        batch_result = processor.run_llm_batch(
            jobs=ready,
            query=query,
            on_progress=on_progress,
        )
        st.session_state.batch_result = batch_result
        st.session_state.is_restored_result = False

        # Persiste no histórico se ao menos um arquivo foi analisado
        if batch_result.successful or batch_result.failed:
            entry = HistoryEntry(
                query=query,
                template=select_prompt_template(query),
                files=[j.filename for j in ready],
                successful=len(batch_result.successful),
                failed=len(batch_result.failed),
                batch_result=batch_result,
            )
            append_entry(st.session_state.query_history, entry)

    except Exception as exc:
        st.error(f"Erro inesperado na análise: {exc}")
    finally:
        progress_bar.empty()
        status_text.empty()

    st.rerun()


# ---------------------------------------------------------------------------
# Resultados do batch
# ---------------------------------------------------------------------------

def render_batch_results() -> None:
    br: BatchResult = st.session_state.batch_result
    if br is None:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📊 Resultados da Análise em Lote")

    # Banner quando o resultado vem do histórico (não é da rodada atual)
    if st.session_state.get("is_restored_result"):
        hist: list[HistoryEntry] = st.session_state.query_history
        origin = next((e for e in hist if e.batch_result is br), None)
        if origin:
            col_info, col_new = st.columns([5, 1])
            col_info.info(
                f"📜 Resultado de **{origin.date_label}** · *\"{origin.query_preview}\"*"
            )
            if col_new.button("✕ Fechar", key="close_restored"):
                st.session_state.batch_result = None
                st.session_state.is_restored_result = False
                st.rerun()

    # Métricas de resumo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arquivos analisados", br.total)
    c2.metric("Bem-sucedidos", len(br.successful))
    c3.metric("Com erro", len(br.failed))
    c4.metric("Taxa de sucesso", f"{br.success_rate * 100:.0f}%")

    # Erros de LLM
    llm_errors = [(j.filename, j.error) for j in br.failed]
    if llm_errors:
        with st.expander(f"❌ {len(llm_errors)} erro(s) na análise"):
            for fname, err in llm_errors:
                st.error(f"**{fname}**: {err}")

    if not br.successful:
        st.warning("Nenhum arquivo foi analisado com sucesso.")
        return

    # Tabs de visualização
    tab_combined, tab_per_file, tab_json = st.tabs(
        ["📊 Combinado", "📁 Por Arquivo", "{ } JSON"]
    )

    with tab_combined:
        _render_combined(br)

    with tab_per_file:
        _render_per_file(br)

    with tab_json:
        st.json(br.to_dict())

    # Downloads
    _render_batch_downloads(br)


def _render_combined(br: BatchResult) -> None:
    """Tabelas mescladas com coluna 'Arquivo' — visão consolidada do lote."""
    combined = br.combined_tables()

    if not combined:
        st.info("Nenhuma tabela extraída para consolidar.")
        _render_free_form_results(br)
        return

    for key, df in combined.items():
        st.markdown(f"**{key.replace('_', ' ').title()}**")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Totalizador de área se aplicável
        area_col = next(
            (c for c in df.columns if c.lower() not in ("arquivo",)
             and ("area" in c.lower() or "m2" in c.lower() or "m²" in c.lower())),
            None,
        )
        if area_col:
            total = pd.to_numeric(df[area_col], errors="coerce").sum()
            if total > 0:
                st.metric(f"Σ {area_col}", f"{total:.2f}")

        st.divider()


def _render_free_form_results(br: BatchResult) -> None:
    """Fallback para consultas que retornam texto (consulta_livre)."""
    for job in br.successful:
        if not job.llm_result:
            continue
        res = job.llm_result.get("resultado")
        if isinstance(res, dict):
            st.markdown(f"**{job.filename}**")
            st.markdown(res.get("descricao", ""))
            dados = res.get("dados")
            if isinstance(dados, list):
                st.dataframe(pd.DataFrame(dados), use_container_width=True)
            elif dados:
                st.json(dados)
            st.divider()


def _render_per_file(br: BatchResult) -> None:
    """Accordion com resultado individual de cada arquivo."""
    for job in br.jobs:
        icon = "✅" if job.is_done else "❌"
        meta = job.llm_result.get("_meta", {}) if job.llm_result else {}
        conf = job.llm_result.get("confianca", "") if job.llm_result else ""
        header = (
            f"{icon} **{job.filename}** · {job.type_badge} · "
            f"parse {job.parse_duration_s:.1f}s · IA {job.llm_duration_s:.1f}s"
        )

        with st.expander(header, expanded=False):
            if job.has_error:
                st.error(job.error)
                continue

            if not job.llm_result:
                st.warning("Sem resultado de IA.")
                continue

            # Confiança
            if conf:
                st.markdown(
                    f"**Confiança:** {_confidence_html(conf)}",
                    unsafe_allow_html=True,
                )

            # Avisos do LLM
            for aviso in job.llm_result.get("avisos", []):
                st.warning(aviso)

            # Tabelas do arquivo
            frames = result_to_dataframe(job.llm_result)
            if frames:
                for tname, df in frames.items():
                    st.markdown(f"*{tname.replace('_', ' ').title()}*")
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                res = job.llm_result.get("resultado")
                if isinstance(res, dict):
                    st.markdown(res.get("descricao", ""))


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _render_batch_downloads(br: BatchResult) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("⬇️ Exportar Lote")

    n_files = len(br.successful)
    combined = br.combined_tables()
    n_sheets = len(combined) + sum(
        len(result_to_dataframe(j.llm_result))
        for j in br.successful if j.llm_result
    )

    col_xl, col_js = st.columns(2)

    with col_xl:
        excel_data = _batch_excel_bytes(br)
        if excel_data:
            st.download_button(
                label=f"📥 Excel — {n_sheets} aba(s) ({n_files} arquivo(s))",
                data=excel_data,
                file_name="lote_projetos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.button("📥 Excel — nada para exportar", disabled=True,
                      use_container_width=True)

    with col_js:
        json_str = json.dumps(br.to_dict(), ensure_ascii=False, indent=2)
        st.download_button(
            label=f"📥 JSON — {n_files} arquivo(s)",
            data=json_str.encode("utf-8"),
            file_name="lote_projetos.json",
            mime="application/json",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = render_sidebar()

    st.title("🏗️ PropTech — Leitor Inteligente de Projetos")
    st.markdown(
        "Faça upload de **um ou mais** arquivos DXF / PDF. "
        "Novos arquivos são parseados incrementalmente — sem re-processar o que já foi carregado."
    )

    render_upload(cfg)

    if st.session_state.jobs:
        render_file_list()
        render_aggregate_stats()
        render_query_section(cfg)

    if st.session_state.batch_result is not None:
        render_batch_results()


if __name__ == "__main__":
    main()
