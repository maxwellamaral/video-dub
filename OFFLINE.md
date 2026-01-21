# Execução Offline - Modelos de IA

## ⚙️ Configuração

O projeto está configurado para **modo offline por padrão** (`OFFLINE_MODE = True` em [src/config.py](src/config.py)).

Isso significa que após baixar os modelos, **não haverá tentativa de conexão com a internet**.

## 📥 Download dos Modelos (Uma Vez)

Para executar o projeto sem conexão à internet, primeiro baixe todos os modelos necessários:

```powershell
# Com conexão à internet, execute:
uv run python download_models.py
```

Este script irá baixar:
- **Whisper Base** (~290 MB) - Transcrição de áudio
- **NLLB-200 Distilled** (~1.2 GB) - Tradução multilíngue
- **MMS-TTS Português** (~100 MB) - Síntese de voz
- **Coqui XTTS v2** (~1.8 GB) - Clonagem de voz (opcional)

**Tempo estimado:** 5-15 minutos (dependendo da conexão)

## 📁 Localização dos Modelos

Os modelos são salvos no cache do Hugging Face:

- **Windows:** `C:\Users\<seu_usuario>\.cache\huggingface\hub`
- **Linux/Mac:** `~/.cache/huggingface/hub`

## 🚀 Execução Offline

Após o download, o projeto funcionará **completamente offline**:

```powershell
# Sem necessidade de internet!
uv run python main_refactored.py
```

## 🔄 Como Funciona

O modo offline é controlado em [src/config.py](src/config.py):

```python
# MODO OFFLINE - Desabilita verificação de internet para modelos Hugging Face
OFFLINE_MODE = True  # Defina como False se quiser permitir downloads automáticos
```

Quando `OFFLINE_MODE = True`:
1. **Define variáveis de ambiente** que bloqueiam tentativas de conexão
2. **Força uso exclusivo** de modelos já em cache
3. **Exibe erro claro** se algum modelo não estiver baixado

## 🌐 Modo Online (Opcional)

Se preferir permitir downloads automáticos quando necessário:

1. Edite [src/config.py](src/config.py)
2. Altere `OFFLINE_MODE = False`
3. Na primeira execução, modelos faltantes serão baixados automaticamente

## ⚠️ Observações

- **Espaço em disco:** Reserve ~3.5 GB para todos os modelos
- **Primeira execução:** Se não executar `download_models.py`, os modelos serão baixados automaticamente na primeira vez (requer internet)
- **Modelos multilíngues:** Para outros idiomas de TTS, execute o projeto uma vez com internet para baixar o modelo específico

## 🧹 Limpeza de Cache (Opcional)

Para remover modelos baixados:

```powershell
# Windows
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub"

# Linux/Mac
rm -rf ~/.cache/huggingface/hub
```

## 📊 Tamanho dos Modelos

| Modelo | Tamanho | Função |
|--------|---------|--------|
| Whisper Base | ~290 MB | Transcrição de áudio |
| NLLB-200 | ~1.2 GB | Tradução |
| MMS-TTS (por) | ~100 MB | Síntese de voz rápida |
| Coqui XTTS v2 | ~1.8 GB | Clonagem de voz (opcional) |
| **Total** | **~3.4 GB** | |
