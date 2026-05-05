"""
Prompt templates para o NLP Processor.
Cada template retorna um par (system_prompt, user_prompt).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema JSON que o LLM deve obedecer em TODAS as respostas
# ---------------------------------------------------------------------------

_JSON_INSTRUCTION = """\
INSTRUÇÕES DE FORMATO (obrigatório):
- Responda SOMENTE com um objeto JSON válido, sem markdown, sem texto adicional.
- Não use blocos ```json```.
- Não adicione comentários dentro do JSON.
- Garanta que todos os campos obrigatórios estejam presentes.
- Se um valor não puder ser determinado, use null (não omita o campo).
"""


# ---------------------------------------------------------------------------
# System prompt base — contexto técnico permanente
# ---------------------------------------------------------------------------

SYSTEM_BASE = """\
Você é um especialista em leitura e interpretação de projetos de arquitetura e \
engenharia civil brasileira. Você recebe dados brutos extraídos de arquivos DXF \
(AutoCAD) ou PDF contendo plantas baixas, cortes e detalhes construtivos.

Seu trabalho é interpretar esses dados com precisão técnica, aplicando \
conhecimento sobre normas brasileiras (ABNT NBR 6492, 9050, 12721) e \
nomenclatura padrão do setor (ambientes, cotas, revestimentos, esquadrias).

""" + _JSON_INSTRUCTION


# ---------------------------------------------------------------------------
# Templates de prompt por tipo de consulta
# ---------------------------------------------------------------------------

def ambientes_e_areas(raw_text: str, user_query: str) -> tuple[str, str]:
    """Extrai tabela de ambientes com área, piso e observações."""
    system = SYSTEM_BASE
    user = f"""\
CONSULTA DO USUÁRIO: {user_query}

DADOS EXTRAÍDOS DO PROJETO:
{raw_text}

Analise os dados acima e retorne um JSON com o seguinte schema:
{{
  "ambientes": [
    {{
      "nome": "string — nome do ambiente (ex: SALA DE ESTAR, DORMITÓRIO 01)",
      "area_m2": "number | null — área em m²",
      "tipo_piso": "string | null — tipo de revestimento do piso",
      "observacoes": "string | null — observações relevantes"
    }}
  ],
  "area_total_m2": "number | null — soma das áreas",
  "unidade_original": "string — unidade das cotas no arquivo (ex: Metros, Milímetros)",
  "confianca": "string — 'alta' | 'media' | 'baixa'",
  "avisos": ["string"]
}}
"""
    return system, user


def quantitativo_esquadrias(raw_text: str, user_query: str) -> tuple[str, str]:
    """Extrai portas, janelas e outros elementos de esquadria."""
    system = SYSTEM_BASE
    user = f"""\
CONSULTA DO USUÁRIO: {user_query}

DADOS EXTRAÍDOS DO PROJETO:
{raw_text}

Analise os dados acima e retorne um JSON com o seguinte schema:
{{
  "esquadrias": [
    {{
      "tipo": "string — PORTA | JANELA | PORTÃO | PORTA_JANELA | BASCULANTE | outro",
      "quantidade": "integer",
      "dimensao": "string | null — ex: '0.90x2.10m', '1.20x1.50m'",
      "material": "string | null — ex: MADEIRA, ALUMÍNIO, PVC",
      "ambiente": "string | null — ambiente onde está instalada",
      "observacoes": "string | null"
    }}
  ],
  "total_portas": "integer",
  "total_janelas": "integer",
  "confianca": "string — 'alta' | 'media' | 'baixa'",
  "avisos": ["string"]
}}
"""
    return system, user


def quantitativo_revestimentos(raw_text: str, user_query: str) -> tuple[str, str]:
    """Extrai quantitativos de piso, parede e teto."""
    system = SYSTEM_BASE
    user = f"""\
CONSULTA DO USUÁRIO: {user_query}

DADOS EXTRAÍDOS DO PROJETO:
{raw_text}

Analise os dados acima e retorne um JSON com o seguinte schema:
{{
  "revestimentos": [
    {{
      "tipo": "string — PISO | PAREDE | TETO",
      "material": "string — ex: CERÂMICA, PORCELANATO, PINTURA, GESSO",
      "ambiente": "string | null",
      "area_m2": "number | null",
      "especificacao": "string | null — dimensão da peça, acabamento, etc."
    }}
  ],
  "confianca": "string — 'alta' | 'media' | 'baixa'",
  "avisos": ["string"]
}}
"""
    return system, user


def consulta_livre(raw_text: str, user_query: str) -> tuple[str, str]:
    """
    Consulta genérica — o LLM interpreta a pergunta e estrutura a resposta.
    Retorna JSON com chave 'resultado' flexível + metadados.
    """
    system = SYSTEM_BASE
    user = f"""\
CONSULTA DO USUÁRIO: {user_query}

DADOS EXTRAÍDOS DO PROJETO:
{raw_text}

Com base nos dados do projeto e na consulta do usuário, retorne um JSON com:
{{
  "consulta": "string — a consulta original parafraseada",
  "resultado": {{
    "descricao": "string — resposta em linguagem técnica clara",
    "dados": "array | object | null — dados estruturados extraídos, se aplicável"
  }},
  "fonte_dos_dados": ["string — quais layers/seções do arquivo embasaram a resposta"],
  "confianca": "string — 'alta' | 'media' | 'baixa'",
  "avisos": ["string — inconsistências, dados faltantes, suposições feitas"]
}}
"""
    return system, user


# ---------------------------------------------------------------------------
# Roteador de templates
# ---------------------------------------------------------------------------

_KEYWORDS_AREA = {"área", "areas", "ambiente", "ambientes", "piso", "cômodo", "comodo", "m²", "m2"}
_KEYWORDS_ESQUADRIA = {"porta", "portas", "janela", "janelas", "esquadria", "esquadrias", "portão"}
_KEYWORDS_REVESTIMENTO = {"revestimento", "revestimentos", "cerâmica", "ceramica", "porcelanato", "piso", "parede", "teto"}


def select_prompt_template(user_query: str) -> str:
    """
    Retorna o nome do template mais adequado com base em palavras-chave da consulta.
    Retorna: 'ambientes_e_areas' | 'quantitativo_esquadrias' |
             'quantitativo_revestimentos' | 'consulta_livre'
    """
    q_lower = user_query.lower()
    words = set(q_lower.split())

    if words & _KEYWORDS_ESQUADRIA:
        return "quantitativo_esquadrias"
    if words & _KEYWORDS_REVESTIMENTO and not words & {"área", "areas", "m²", "m2"}:
        return "quantitativo_revestimentos"
    if words & _KEYWORDS_AREA:
        return "ambientes_e_areas"
    return "consulta_livre"


TEMPLATE_REGISTRY: dict[str, callable] = {
    "ambientes_e_areas": ambientes_e_areas,
    "quantitativo_esquadrias": quantitativo_esquadrias,
    "quantitativo_revestimentos": quantitativo_revestimentos,
    "consulta_livre": consulta_livre,
}
