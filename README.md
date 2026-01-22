# Dubbler Pro - Pipeline de Dublagem Automática (v2.0)

Sistema automatizado para dublagem de vídeos utilizando Inteligência Artificial. O projeto realiza transcrição, tradução, síntese de voz (TTS) e sincronização labial (em desenvolvimento/sincronia temporal), tudo otimizado para GPUs NVIDIA.

---

## 🚀 Features

- **Arquitetura Modular**: Código organizado em serviços independentes (`src/services/`) para fácil manutenção.
- **Download de Vídeos do YouTube**: Baixe vídeos diretamente do YouTube para processamento (novo!).
- **Múltiplos Motores TTS**:
  - **MMS-TTS (Facebook)**: Rápido, leve e totalmente offline.
  - **Coqui XTTS v2**: Alta qualidade com clonagem de voz (Voice Cloning) a partir do vídeo original.
- **Encoding Inteligente**:
  - **Modo Rápido**: Aceleração via GPU (`h264_nvenc`).
  - **Modo Qualidade**: Compressão superior via CPU (`libx264`) com correção automática de áudio.
- **Resiliência**: Tratamento robusto de erros (WinError 6, falhas de I/O) e limpeza automática de recursos.
- **Testes Automatizados**: Suíte completa (`pytest`) para validar o pipeline.

## 🛠️ Arquitetura do Projeto

O sistema foi refatorado para seguir boas práticas de Engenharia de Software:

```
video-dub/
├── run_app.ps1             # Inicializador da interface web (Windows)
├── run_app.sh              # Inicializador da interface web (Linux/macOS)
├── pyproject.toml          # Configuração do projeto e dependências (uv)
├── tests/                  # Testes Automatizados (pytest)
└── src/                    # Código Fonte Modular
    ├── main.py             # Ponto de Entrada CLI (Entrypoint)
    ├── download_models.py  # Script para download de modelos offline
    ├── config.py           # Configurações Globais (Caminhos, GPU, Modo Offline)
    ├── pipeline.py         # Orquestrador Principal
    ├── utils.py            # Funções Auxiliares (FFmpeg helper, logs)
    ├── services/           # Serviços Especializados de IA
    │   ├── audio.py        # Extração de Áudio e Transcrição (Whisper)
    │   ├── translation.py  # Tradução Neural (NLLB)
    │   ├── tts.py          # Síntese de Voz (MMS/Coqui)
    │   └── video.py        # Sincronização e Renderização (MoviePy)
    ├── backend/            # API FastAPI para interface web
    │   └── app.py          # Endpoints e WebSocket para progresso
    └── frontend/           # Interface Vue.js
        ├── src/            # Componentes Vue
        └── package.json    # Dependências do frontend
```

## 📋 Pré-requisitos

- **Python**: 3.11 (gerenciado pelo uv)
- **FFmpeg**: Instalado e acessível no PATH (o script tenta detectar automaticamente)
- **GPU NVIDIA** (Opcional, mas recomendado): Para transcrição Whisper e codec NVENC
- **CUDA Toolkit**: 12.4 (configurado automaticamente com PyTorch)
- **Node.js**: Para executar a interface web (opcional)

## 📦 Instalação

### 1. Instalar o uv (Gerenciador de Pacotes Python)

O projeto usa o [uv](https://github.com/astral-sh/uv), um gerenciador de pacotes Python extremamente rápido escrito em Rust.

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verificar instalação:**

```bash
uv --version
```

### 2. Clonar e Configurar o Projeto

1. Clone o repositório e entre na pasta.
2. Instale as dependências com `uv`:
   ```bash
   uv sync
   ```
   _Nota: O projeto usa Python 3.11 e PyTorch com CUDA 12.4 configurados automaticamente._

**O que o `uv sync` faz?**

- Cria automaticamente um ambiente virtual em `.venv/`
- Instala o Python 3.11 se necessário
- Instala todas as dependências do `pyproject.toml`
- Configura o PyTorch com suporte a CUDA 12.4

### 📥 Download de Modelos para Execução Offline (Recomendado)

Para usar o projeto sem conexão à internet, baixe os modelos uma vez:

```bash
uv run python src/download_models.py
```

Isso baixará ~3.4 GB de modelos de IA. Depois, o projeto funcionará completamente offline!

📖 **Mais detalhes:** Veja [OFFLINE.md](OFFLINE.md)

## ▶️ Como Usar

### 1. Preparação

Coloque o vídeo que deseja dublar na pasta `input/` e renomeie para `video_entrada.mp4` (ou ajuste no menu).

### 2. Execução CLI

Execute o arquivo principal:

```bash
uv run python src/main.py
```

Siga o menu interativo:

1. Escolha o motor de voz (MMS ou Coqui).
2. Escolha o modo de encoding (Rápido/GPU ou Qualidade/CPU).

O resultado será salvo na pasta `output/` como `video_dublado_{motor}.mp4`.

### 3. Interface Web (Novo!)

Para uma experiência visual com logs em tempo real:

**Windows (PowerShell):**

```powershell
.\run_app.ps1
```

**Linux/macOS (Bash):**

```bash
chmod +x run_app.sh  # Primeira vez apenas
./run_app.sh
```

Isso iniciará o backend (FastAPI) e frontend (Vue.js) em segundo plano.

2. Acesse no navegador:
   `http://localhost:5173`

3. Na interface:
   - Faça upload do vídeo.
   - Escolha o Motor (MMS/Coqui).
   - Acompanhe o progresso no terminal embutido.
   - Baixe o vídeo final diretamente da página.

**Logs:** Os logs são salvos em `logs/backend.log` e `logs/frontend.log`

3. Na interface:
   - Faça upload do vídeo.
   - Escolha o Motor (MMS/Coqui).
   - Acompanhe o progresso no terminal embutido.
   - Baixe o vídeo final diretamente da página.

### 4. Download de Vídeos do YouTube (Novo!)

A interface web agora suporta download direto de vídeos do YouTube para processamento.

**Como usar:**

1. Na interface web (`http://localhost:5173`), selecione a aba **"YouTube URL"**.

2. Cole a URL do vídeo do YouTube:
   - Formatos aceitos:
     - `https://www.youtube.com/watch?v=VIDEO_ID`
     - `https://youtu.be/VIDEO_ID`
     - `https://www.youtube.com/embed/VIDEO_ID`

3. Clique em **"Baixar do YouTube"**.

4. Aguarde o download (progresso será exibido no terminal).

5. Após o download, selecione o motor TTS e modo de encoding.

6. Clique em **"Iniciar Dublagem"** para processar o vídeo.

**Exemplos de URLs válidas:**

```
https://www.youtube.com/watch?v=jNQXAC9IVRw
https://youtu.be/dQw4w9WgXcQ
```

**Limitações:**

- Apenas vídeos públicos podem ser baixados.
- Vídeos com restrições geográficas podem falhar.
- Vídeos privados ou removidos não são acessíveis.
- A qualidade máxima de download é 1080p.

> [!WARNING]
> **Direitos Autorais**: Certifique-se de ter permissão para baixar e processar o vídeo. Esta ferramenta destina-se apenas a fins educacionais e de pesquisa. Respeite as leis de direitos autorais aplicáveis.

## 🧪 Testes

Para verificar a integridade da instalação e do pipeline, execute a suíte de testes:

```powershell
python -m pytest tests/ -v
```

Os testes validam:

- Detecção de ambiente (CUDA, FFmpeg).
- Pipeline MMS (End-to-end com vídeo sintético).
- Pipeline Coqui (Carregamento e execução básica).

## ⚠️ Solução de Problemas Comuns

- **WinError 6 (Invalid Handle)**: Geralmente causado por antivírus ou delay de sistema de arquivos. O script possui retry automático.
- **Vídeo sem Áudio**: Use o modo "Qualidade" ou garanta que o FFmpeg esteja atualizado. O script força muxing de áudio `aac` para compatibilidade.
- **Accessing time... Error**: Erro de ponto flutuante do MoviePy corrigido nesta versão via padding de áudio.

## 📚 Citação

Se você usar este projeto em sua pesquisa ou trabalho acadêmico, por favor cite:

```bibtex
@software{amaral2026videodub,
  author       = {Maxwell Anderson Ielpo do Amaral},
  title        = {Video Dubbing System: AI-Powered Automatic Video Dubbing with Voice Cloning},
  year         = {2026},
  publisher    = {GitHub},
  version      = {0.1.0},
  url          = {https://github.com/maxwellamaral/32-31-video-dub},
  note         = {Sistema de dublagem automática de vídeos com IA utilizando Whisper, NLLB-200 e TTS}
}
```

Ou consulte o arquivo [CITATION.bib](CITATION.bib).

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

_Desenvolvido com foco em automação e qualidade via Python._
