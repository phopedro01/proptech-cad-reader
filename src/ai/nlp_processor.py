"""
NLP Processor — Integração LLM
================================
Recebe o texto bruto extraído pelo parser DXF/PDF,
monta prompts estruturados e retorna JSON interpretado.

Suporta: OpenAI (gpt-4o, gpt-4-turbo) e Google Gemini (gemini-1.5-pro).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config.settings import settings
from src.ai.prompts import TEMPLATE_REGISTRY, select_prompt_template

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex para extrair JSON de respostas que eventualmente tenham markdown
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    """
    Tenta parsear JSON da resposta do LLM.
    Suporta resposta pura ou envolta em bloco ```json```.
    """
    text = text.strip()

    # tenta direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # tenta extrair de bloco markdown
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Resposta do LLM não é JSON válido.\nResposta recebida:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Clientes LLM
# ---------------------------------------------------------------------------

def _build_openai_client():
    """Retorna um ChatOpenAI (LangChain wrapper)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
    )


def _build_gemini_client():
    """Retorna um ChatGoogleGenerativeAI (LangChain wrapper)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_TOKENS,
    )


def _get_llm():
    """Factory — retorna o cliente correto com base em LLM_PROVIDER."""
    provider = settings.LLM_PROVIDER
    if provider == "openai":
        return _build_openai_client()
    if provider == "gemini":
        return _build_gemini_client()
    raise ValueError(
        f"LLM_PROVIDER inválido: '{provider}'. Use 'openai' ou 'gemini'."
    )


# ---------------------------------------------------------------------------
# Processador principal
# ---------------------------------------------------------------------------

class NLPProcessor:
    """
    Interpreta dados brutos de projetos de engenharia via LLM.

    Uso típico:
        processor = NLPProcessor()
        result = processor.process(
            raw_text=dxf_result.to_text_summary(),
            user_query="Extraia uma tabela com todos os ambientes e suas áreas"
        )
        print(result["ambientes"])
    """

    def __init__(self, llm=None) -> None:
        # permite injeção de dependência (útil em testes)
        self._llm = llm or _get_llm()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def process(
        self,
        raw_text: str,
        user_query: str,
        template_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Envia o texto extraído ao LLM e retorna um dicionário estruturado.

        Parâmetros
        ----------
        raw_text : str
            Texto gerado pelo DXFParseResult.to_text_summary() ou PDF parser.
        user_query : str
            Pergunta do usuário em linguagem natural.
            Ex: "Extraia uma tabela com todos os ambientes e suas áreas"
        template_name : str | None
            Força um template específico. Se None, é inferido automaticamente
            pelas palavras-chave da consulta.

        Retorna
        -------
        dict contendo os dados estruturados + metadados (confianca, avisos).

        Raises
        ------
        ValueError  : resposta não parseável como JSON após 3 tentativas.
        """
        if not raw_text.strip():
            raise ValueError("raw_text está vazio — nada a processar.")
        if not user_query.strip():
            raise ValueError("user_query está vazio.")

        # Seleciona template
        chosen = template_name or select_prompt_template(user_query)
        template_fn = TEMPLATE_REGISTRY.get(chosen, TEMPLATE_REGISTRY["consulta_livre"])
        system_prompt, user_prompt = template_fn(raw_text, user_query)

        logger.info("NLPProcessor: template=%s | provider=%s", chosen, settings.LLM_PROVIDER)

        # Chama LLM via LangChain (interface unificada)
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self._llm.invoke(messages)
        raw_response: str = response.content

        logger.debug("Resposta bruta LLM (primeiros 500 chars): %s", raw_response[:500])

        result = _extract_json(raw_response)
        result["_meta"] = {
            "template": chosen,
            "llm_provider": settings.LLM_PROVIDER,
            "llm_model": (
                settings.OPENAI_MODEL
                if settings.LLM_PROVIDER == "openai"
                else settings.GEMINI_MODEL
            ),
        }
        return result

    def process_batch(
        self,
        raw_text: str,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """
        Executa múltiplas consultas sobre o mesmo documento.

        Útil para gerar em um passo: ambientes + esquadrias + revestimentos.
        """
        results = []
        for query in queries:
            try:
                results.append(self.process(raw_text=raw_text, user_query=query))
            except Exception as exc:
                logger.error("Erro na consulta '%s': %s", query[:80], exc)
                results.append({"erro": str(exc), "consulta": query})
        return results
