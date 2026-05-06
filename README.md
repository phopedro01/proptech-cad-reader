# PropTech — Leitor Inteligente de Projetos de Engenharia Civil

[![GitHub](https://img.shields.io/badge/GitHub-phopedro01%2Fproptech--cad--reader-blue?logo=github)](https://github.com/phopedro01/proptech-cad-reader)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/phopedro01/proptech-cad-reader/blob/master/LICENSE)

Extração automática de dados estruturados a partir de arquivos **DXF** (AutoCAD) e **PDF** de projetos de arquitetura e engenharia civil, com análise via **LLM** (OpenAI / Google Gemini) e interface **Streamlit**.

---

## Funcionalidades

- **Parser DXF** — extrai textos, blocos (portas, janelas), cotas, polilínhas fechadas com área calculada e hachuras de piso via `ezdxf`
- **Parser PDF** — modo digital (PyMuPDF + pdfplumber) com fallback automático para OCR (Tesseract + OpenCV) em PDFs escaneados
- **Análise com IA** — prompts estruturados entregam JSON com ambientes, áreas, esquadrias e revestimentos; suporte a OpenAI e Google Gemini via LangChain
- **Batch** — processa múltiplos arquivos em lote com cache incremental por hash MD5 (novos arquivos não re-parseiam os já carregados)
- **Histórico de sessão** — últimas 20 consultas acessíveis na sidebar com opção de rever resultados anteriores
- **Exportação** — Excel multi-aba (`COMB_` + por arquivo) e JSON completo

---

## Estrutura do Projeto

```
app/
├── app.py                        # Interface Streamlit
├── requirements.txt
├── .env.example
├── pytest.ini
├── config/
│   └── settings.py               # Configurações centralizadas via .env
└── src/
    ├── parsers/
    │   ├── dwg_parser.py         # Parser DXF (ezdxf)
    │   └── pdf_parser.py         # Parser PDF (PyMuPDF + OCR)
    ├── batch/
    │   └── batch_processor.py    # Pipeline de batch com cache MD5
    ├── ai/
    │   ├── prompts.py            # Templates de prompt por tipo de consulta
    │   └── nlp_processor.py      # Integração LLM via LangChain
    ├── history.py                # Histórico de consultas por sessão
    └── utils/
        └── exporters.py          # Exportação Excel e JSON
tests/
    ├── test_dwg_parser.py
    ├── test_pdf_parser.py
    ├── test_batch_processor.py
    ├── test_history.py
    └── integration/
        ├── conftest.py
        └── test_batch_flow.py    # 46 testes de integração
```

---

## Pré-requisitos

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (opcional — apenas para PDFs escaneados)
- Chave de API OpenAI **ou** Google Gemini

> **DWG**: `ezdxf` suporta apenas DXF. Converta arquivos `.dwg` com [ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter) (gratuito), AutoCAD ou FreeCAD.

---

## Instalação

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\Activate.ps1       # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar credenciais
cp .env.example .env
# Editar .env com sua OPENAI_API_KEY ou GOOGLE_API_KEY
```

---

## Configuração

Edite o arquivo `.env`:

```env
# Provedor LLM: openai | gemini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# ou Google Gemini
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini-1.5-pro

# Tesseract (Windows)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Uso

### Interface Streamlit

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`:

1. Faça upload de um ou mais arquivos DXF / PDF
2. Aguarde o parse automático (barra de progresso por arquivo)
3. Digite ou selecione uma consulta pré-definida
4. Clique em **Analisar em Lote** — resultados aparecem em tabelas combinadas e por arquivo
5. Baixe em Excel ou JSON

### Uso programático

```python
from src.parsers.dwg_parser import DXFParser
from src.ai.nlp_processor import NLPProcessor

# Parse
result = DXFParser().parse("planta.dxf")

# Análise
dados = NLPProcessor().process(
    raw_text=result.to_text_summary(),
    user_query="Extraia todos os ambientes e suas áreas em m²"
)

print(dados["ambientes"])
```

---

## Templates de Consulta

| Template | Consultas típicas |
|---|---|
| `ambientes_e_areas` | "Extraia ambientes e áreas", "lista de cômodos com m²" |
| `quantitativo_esquadrias` | "Quantas portas e janelas?", "liste as esquadrias" |
| `quantitativo_revestimentos` | "Quais os tipos de piso?", "revestimentos por ambiente" |
| `consulta_livre` | Qualquer pergunta técnica sobre o projeto |

O template é selecionado automaticamente por palavras-chave ou pode ser forçado via parâmetro.

---

## Testes

```bash
# Suite completa (119 testes)
pytest tests/ -v

# Apenas unitários
pytest tests/ -v -m "not integration"

# Apenas integração
pytest tests/ -v -m integration

# Com cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| DXF/DWG | `ezdxf` |
| PDF digital | `PyMuPDF`, `pdfplumber` |
| OCR | `pytesseract`, `OpenCV`, `Pillow` |
| IA / LLM | `LangChain`, `langchain-openai`, `langchain-google-genai` |
| Interface | `Streamlit` |
| Dados | `pandas`, `openpyxl` |
| Testes | `pytest`, `pytest-cov` |

---

## Links

- **Repositório:** https://github.com/phopedro01/proptech-cad-reader
- **Issues:** https://github.com/phopedro01/proptech-cad-reader/issues
- **Releases:** https://github.com/phopedro01/proptech-cad-reader/releases

---

## Licença

MIT — veja o arquivo [LICENSE](LICENSE) para detalhes.
