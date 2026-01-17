# GUIA DE INSTALAÇÃO E USO - PIPELINE DE DUBLAGEM LOCAL
## Com Hugging Face Transformers em GPU NVIDIA

---

## 📋 PRÉ-REQUISITOS

### Hardware
- **GPU NVIDIA** com pelo menos 4-6 GB de VRAM (ideal 8GB+)
- **CPU** com processador moderno
- **Armazenamento** de ~30 GB para modelos baixados

### Sistema Operacional
- Linux/WSL2 (recomendado) ou Windows nativo
- Conexão com a internet (para baixar modelos)

### Software Base
```bash
# Se no WSL2 ou Linux, instale ffmpeg (necessário para manipular vídeos)
sudo apt-get update
sudo apt-get install ffmpeg

# Windows: baixe de https://ffmpeg.org/download.html
# Ou use: choco install ffmpeg (se usar Chocolatey)
```

---

## 🔧 INSTALAÇÃO DO AMBIENTE PYTHON

### 1. Criar Ambiente Virtual
```bash
# Criar ambiente
python3 -m venv venv_dublagem

# Ativar
# Linux/WSL:
source venv_dublagem/bin/activate

# Windows:
venv_dublagem\Scripts\activate
```

### 2. Instalar Dependências
```bash
# Atualizar pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Instalar PyTorch com suporte CUDA
# Para CUDA 12.1 (compatível com drivers NVIDIA recentes):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Ou para CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar transformers, datasets e librosa
pip install transformers datasets librosa soundfile

# Opcional: para melhor performance
pip install flash-attn  # Atenção otimizada
```

### 3. Verificar Instalação
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Nenhuma\"}')"
```

Deve retornar algo como:
```
PyTorch: 2.1.0+cu121
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3060
```

---

## 🚀 USO DO SCRIPT DE DUBLAGEM

### 1. Preparar Vídeo
```bash
# Coloque seu vídeo no mesmo diretório do script e renomeie para:
# video_entrada.mp4
```

### 2. Configurar Idiomas (Opcional)
Abra `pipeline_dublagem.py` e edite essas linhas conforme necessário:

```python
# Idioma original do vídeo
IDIOMA_ORIGEM = "en_XX"      # en_XX = inglês, es_ES = espanhol, etc.

# Idioma para tradução e dublagem
IDIOMA_DESTINO = "pt_BR"     # pt_BR = português brasileiro
IDIOMA_VOZ = "pt"            # Idioma da síntese de voz
```

**Códigos de idiomas suportados:**
- Inglês: `en_XX`
- Português Brasileiro: `pt_BR`
- Português Europeu: `pt_PT`
- Espanhol: `es_ES`
- Francês: `fr_XX`
- Alemão: `de_DE`
- Italiano: `it_IT`
- Japonês: `ja_XX`
- Chinês: `zh_Hans` ou `zh_Hant`
- Russo: `ru_RU`
- Árabe: `ar_AR`
- [+ 190+ idiomas suportados pelo NLLB]

### 3. Executar Pipeline
```bash
python pipeline_dublagem.py
```

Você verá algo como:
```
✓ Usando dispositivo: cuda:0
======================================================================
INICIANDO PIPELINE DE DUBLAGEM
======================================================================

📹 Extraindo áudio de: video_entrada.mp4
✓ Áudio extraído: audio_extraido.wav

🎙️  Transcrevendo áudio com Whisper...
(Isso pode levar alguns minutos na primeira vez)
✓ Texto transcrito (245 caracteres):
   "Hello, this is a test video for dubbing with artificial intelligence..."

🌐 Traduzindo de en_XX para pt_BR...
✓ Texto traduzido (280 caracteres):
   "Olá, este é um vídeo de teste para dublagem com inteligência artificial..."

🔊 Sintetizando fala em pt...
✓ Áudio sintetizado (88200 amostras)
✓ Áudio salvo em: audio_traduzido.wav

🎬 Remontando vídeo com áudio dublado...
✓ Vídeo dublado salvo em: video_dublado.mp4

======================================================================
✓ PIPELINE COMPLETADA COM SUCESSO!
✓ Vídeo dublado salvo em: video_dublado.mp4
======================================================================
```

### 4. Acessar Resultado
O vídeo dublado estará em `video_dublado.mp4` no mesmo diretório.

---

## ⚙️ AJUSTES POR PERFORMANCE

### Se ficar SEM MEMÓRIA GPU:

**Opção 1: Usar modelo Whisper menor**
```python
# Trocar de:
model="openai/whisper-base"

# Para:
model="openai/whisper-tiny"  # Mais rápido, menos preciso
# ou
model="openai/whisper-small"  # Bom balanço
```

**Opção 2: Usar float16 em vez de float32**
Já está configurado por padrão no script.

**Opção 3: Usar modelo NLLB menor**
```python
# Trocar de:
model="facebook/nllb-200-distilled-600M"

# Para:
model="facebook/nllb-200-distilled-600M"  # Já é o menor
# Se precisar de mais velocidade, usar modelo de 1.3B é o próximo salto
```

**Opção 4: Processar vídeo em chunks**
Se o vídeo for muito longo (>30 min), divida em pedaços:
```bash
ffmpeg -i video_entrada.mp4 -c copy -segment_time 5m -f segment "chunk_%03d.mp4"
```

---

## 🔊 PERSONALIZAÇÕES AVANÇADAS

### 1. Modificar Velocidade da Fala

Encontre a função `sintetizar_voz()` e adicione:
```python
# Antes de gerar, ajustar duração
inputs = tokenizer(text=texto, return_tensors="pt").to(DEVICE)

# Aumentar duração em 20%
# (requer modelo com suporte a duração estocástica)
```

### 2. Usar Modelo TTS Alternativo (Parler-TTS)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

modelo_parler = "parler-tts/parler_tts_mini_v1"
tokenizer_parler = AutoTokenizer.from_pretrained(modelo_parler)
model_parler = AutoModelForCausalLM.from_pretrained(modelo_parler).to(DEVICE)

# Gerar fala com descrição de speaker
description = "A 22 year old woman with a slightly high-pitched voice speaks clearly"
```

### 3. Adicionar Sincronização de Lábios (Futura)

Requer modelo adicional como `wav2lip` ou similares.

---

## 🐛 RESOLUÇÃO DE PROBLEMAS

### Erro: "ffmpeg not found"
```bash
# Linux/WSL:
sudo apt-get install ffmpeg

# Windows (Chocolatey):
choco install ffmpeg

# Windows (Manual):
Baixe de https://ffmpeg.org/download.html e adicione ao PATH
```

### Erro: "CUDA out of memory"
→ Use `model="openai/whisper-tiny"` no lugar de `base` ou `small`

### Erro: "No module named 'torch'"
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Tradução com qualidade ruim
→ Tente o modelo completo `facebook/nllb-200-1.3B` em vez de `-distilled-600M`
(Requer mais VRAM, ~6GB)

### Voz de síntese muito robótica
→ Considere usar `parler-tts` ou `bark` para maior naturalidade
(Mais pesados em recursos)

---

## 📚 PRÓXIMOS PASSOS

### Melhorias Recomendadas:
1. **Sincronização Labial**: Integrar `wav2lip` para sincronizar movimento labial
2. **Múltiplas Vozes**: Detectar falantes e manter identidades de voz
3. **Processamento em Batch**: Dividir vídeos longos automaticamente
4. **Interface Web**: Usar Gradio ou Streamlit para tornar mais amigável
5. **Cache de Modelos**: Evitar redownload de modelos já baixados

### Repositórios Úteis:
- [Transformers Hugging Face](https://github.com/huggingface/transformers)
- [SoniTranslate](https://github.com/R3gm/SoniTranslate)
- [Bark TTS](https://github.com/suno-ai/bark)
- [Wav2Lip](https://github.com/justinzhao/Wav2Lip_288)

---

## 📞 RECURSOS ADICIONAIS

- **Documentação Transformers**: https://huggingface.co/docs/transformers/
- **Modelos Disponíveis**: https://huggingface.co/models
- **Problemas/Issues**: https://github.com/huggingface/transformers/issues

---

## ⚖️ AVISOS LEGAIS

- Respeite direitos autorais ao dublar conteúdo de terceiros
- Para fins comerciais, obtenha permissão do criador original
- As vozes sintetizadas podem ser detectadas como IA em algumas plataformas

