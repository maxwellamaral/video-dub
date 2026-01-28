# Documentação: Análise de Emoções no Pipeline de Dublagem

## Visão Geral

O sistema de dublagem de vídeos foi aprimorado com **análise de emoções** utilizando o modelo **SenseVoiceSmall** da FunAudioLLM. Esta funcionalidade detecta automaticamente as emoções presentes no áudio original e as utiliza para gerar uma dublagem mais expressiva e natural com o **Qwen3-TTS**.

## Arquitetura do Sistema

### Pipeline Completo com Emoções

```
┌─────────────────────────────────────────────────────────────────┐
│                     VÍDEO DE ENTRADA                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. EXTRAÇÃO DE ÁUDIO (FFmpeg)                                  │
│     └─> audio_extraido.wav                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. TRANSCRIÇÃO + ANÁLISE DE EMOÇÕES                           │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ 2a. Whisper: Transcrição com Timestamps             │   │
│     │     └─> Texto + Start/End por segmento              │   │
│     └──────────────────────────────────────────────────────┘   │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ 2b. SenseVoice: Detecção de Emoções                 │   │
│     │     └─> Por segmento: angry, happy, sad, neutral... │   │
│     └──────────────────────────────────────────────────────┘   │
│                                                                  │
│  Resultado: Segmentos com texto + timestamps + emoção           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. TRADUÇÃO (NLLB)                                             │
│     └─> Traduz texto preservando emoções detectadas            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. SÍNTESE TTS COM EMOÇÕES (Qwen3-TTS)                        │
│     ┌──────────────────────────────────────────────────────┐   │
│     │ Para cada segmento:                                  │   │
│     │   - Texto traduzido                                  │   │
│     │   - Instrução emocional baseada na emoção detectada │   │
│     │     Exemplo: "Fale com tom alegre e entusiasmado,   │   │
│     │              voz animada, transmitindo felicidade"   │   │
│     └──────────────────────────────────────────────────────┘   │
│                                                                  │
│  Resultado: Áudio dublado com expressividade emocional          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. MONTAGEM FINAL (MoviePy)                                   │
│     └─> Vídeo dublado + Legendas com tags de emoção           │
└─────────────────────────────────────────────────────────────────┘
```

## Componentes do Sistema

### 1. SenseVoiceSmall (Detecção de Emoções)

**Arquivo:** [`src/services/emotion.py`](../src/services/emotion.py)

**Classe Principal:** `EmotionAnalyzer`

#### Funcionalidades:
- Detecta emoções em segmentos de áudio
- Suporta 7 emoções: neutral, happy, sad, angry, fearful, disgusted, surprised
- Retorna tags de emoção e instruções detalhadas para TTS

#### Uso:
```python
from src.services.emotion import EmotionAnalyzer

analyzer = EmotionAnalyzer(modelo="FunAudioLLM/SenseVoiceSmall")
segmentos_com_emocao = analyzer.analisar_audio(
    caminho_audio="audio.wav",
    segmentos=segmentos_transcritos
)
```

#### Formato de Saída:
Cada segmento é enriquecido com:
```python
{
    "start": 0.5,
    "end": 3.2,
    "text": "Hello, how are you?",
    "emotion": "happy",                    # Código da emoção
    "emotion_pt": "feliz",                 # Emoção em português
    "emotion_instruction": "Fale com tom alegre..."  # Instrução para TTS
}
```

### 2. Integração com Whisper

**Arquivo:** [`src/services/audio.py`](../src/services/audio.py)

**Função Principal:** `transcrever_com_emocao()`

Combina Whisper (transcrição) + SenseVoice (emoções) em uma única chamada:

```python
from src.services.audio import transcrever_com_emocao

segmentos = transcrever_com_emocao(
    caminho_audio="audio.wav",
    modelo_whisper="openai/whisper-base",
    modelo_sensevoice="FunAudioLLM/SenseVoiceSmall"
)
# Retorna segmentos com texto + timestamps + emoções
```

### 3. Mapeamento de Emoções para Qwen3-TTS

**Arquivo:** [`src/services/emotion.py`](../src/services/emotion.py)

As emoções detectadas são convertidas em **instruções em linguagem natural** que o Qwen3-TTS compreende:

```python
EMOTION_INSTRUCTIONS = {
    "angry": "Fale com tom zangado, voz elevada e ritmo acelerado, demonstrando irritação",
    "happy": "Fale com tom alegre e entusiasmado, voz animada, transmitindo felicidade",
    "sad": "Fale com tom triste e melancólico, voz baixa e lenta, demonstrando tristeza",
    "neutral": "Fale com tom neutro e equilibrado, voz clara e natural",
    # ... outras emoções
}
```

### 4. Síntese TTS com Emoções

**Arquivo:** [`src/services/tts.py`](../src/services/tts.py)

**Classe:** `TTSEngine`

O motor TTS foi modificado para aceitar instruções emocionais por segmento:

```python
tts = TTSEngine(motor="qwen3", qwen3_mode="custom", qwen3_speaker="vivian")

# Entrada com emoções
textos = [
    {
        "text": "Olá, como você está?",
        "emotion_instruction": "Fale com tom alegre e entusiasmado..."
    },
    {
        "text": "Estou muito cansado.",
        "emotion_instruction": "Fale com tom triste e melancólico..."
    }
]

audios = tts.sintetizar_batch(textos)
```

#### Modos do Qwen3-TTS:

1. **CustomVoice** (padrão): Usa speakers pré-definidos + instrução emocional
2. **VoiceDesign**: Cria voz baseada em descrição livre + emoção
3. **Clone**: Clona voz de referência + aplica emoção

### 5. Legendas com Tags de Emoção

**Arquivo:** [`src/utils.py`](../src/utils.py)

**Função:** `segmentos_para_srt_com_emocao()`

Gera legendas SRT com tags de emoção:

```srt
1
00:00:01,000 --> 00:00:05,500
[FELIZ] Olá, como você está?

2
00:00:06,000 --> 00:00:10,200
[TRISTE] Estou muito cansado hoje...

3
00:00:11,000 --> 00:00:15,800
Este é um diálogo neutro sem tag.
```

**Nota:** Segmentos com emoção "neutral" não recebem tag para evitar poluição visual.

## Configuração

**Arquivo:** [`src/config.py`](../src/config.py)

### Variáveis de Configuração:

```python
# Habilitar/Desabilitar análise de emoções
ENABLE_EMOTION_ANALYSIS = True  # False desativa detecção de emoções

# Incluir tags nas legendas
INCLUDE_EMOTION_TAGS_IN_SUBTITLES = True  # [FELIZ], [TRISTE], etc.

# Modelo SenseVoice
SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"

# Emoções suportadas
SUPPORTED_EMOTIONS = [
    "neutral", "happy", "sad", "angry",
    "fearful", "disgusted", "surprised"
]
```

## Fluxo de Dados Completo

### Entrada:
```
Vídeo em inglês com áudio emocional (pessoa falando com raiva, alegria, etc.)
```

### Processamento:

1. **Extração de Áudio:**
   ```
   video.mp4 → audio_extraido.wav
   ```

2. **Transcrição com Whisper:**
   ```
   "I can't believe this happened!"
   [start: 0.5s, end: 2.3s]
   ```

3. **Detecção de Emoção com SenseVoice:**
   ```
   Emoção detectada: "angry" (zangado)
   Instrução: "Fale com tom zangado, voz elevada..."
   ```

4. **Tradução:**
   ```
   "Não posso acreditar que isso aconteceu!"
   [mantém emoção: "angry"]
   ```

5. **Síntese TTS com Qwen3:**
   ```
   Input: texto="Não posso acreditar que isso aconteceu!"
          speaker="vivian"
          instruct="Fale com tom zangado, voz elevada..."
   
   Output: áudio_pt_zangado.wav
   ```

6. **Legendas Finais:**
   ```srt
   1
   00:00:00,500 --> 00:00:02,300
   [ZANGADO] Não posso acreditar que isso aconteceu!
   ```

### Saída:
```
Vídeo dublado em português com:
- Áudio sintetizado com expressividade emocional apropriada
- Legendas com tags de emoção
```

## Estatísticas de Emoções

O pipeline também gera estatísticas sobre as emoções detectadas:

```python
from src.utils import extrair_estatisticas_emocoes

stats = extrair_estatisticas_emocoes(segmentos)
# {
#     "total": 50,
#     "emocoes": {"happy": 20, "sad": 10, "neutral": 15, "angry": 5},
#     "predominante": "happy",
#     "distribuicao_percentual": {
#         "happy": 40.0,
#         "sad": 20.0,
#         "neutral": 30.0,
#         "angry": 10.0
#     }
# }
```

Exemplo de saída no log:
```
📊 Estatísticas de Emoções:
   Total de segmentos: 50
   Emoção predominante: happy
   - happy: 20 (40.0%)
   - sad: 10 (20.0%)
   - neutral: 15 (30.0%)
   - angry: 5 (10.0%)
```

## Exemplos de Uso

### Exemplo 1: Pipeline Completo com Emoções

```python
from src.pipeline import executar_pipeline

sucesso = executar_pipeline(
    caminho_video="input/video.mp4",
    idioma_origem="eng_Latn",
    idioma_destino="por_Latn",
    idioma_voz="por",
    motor_tts="qwen3",
    modo_encoding="qualidade",
    qwen3_mode="custom",
    qwen3_speaker="vivian"
)
```

### Exemplo 2: Apenas Análise de Emoções

```python
from src.services.emotion import EmotionAnalyzer

analyzer = EmotionAnalyzer()
resultado = analyzer.analisar_audio("audio.wav")

print(f"Emoção: {resultado['emotion']}")
print(f"Instrução: {resultado['instruction']}")
```

### Exemplo 3: Desabilitar Análise de Emoções

Em `src/config.py`:
```python
ENABLE_EMOTION_ANALYSIS = False  # Volta ao pipeline original sem emoções
```

## Requisitos de Sistema

### Modelos Necessários:

1. **Whisper** (transcrição)
   - `openai/whisper-base` (padrão)
   - Outros: whisper-small, whisper-medium, whisper-large

2. **SenseVoiceSmall** (emoções)
   - `FunAudioLLM/SenseVoiceSmall`
   - ~2GB de espaço em disco

3. **Qwen3-TTS** (síntese)
   - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (padrão)
   - ~3.4GB de espaço em disco

### Instalação:

```bash
# Instalar dependências
uv sync

# Download dos modelos (modo offline)
python scripts/download_models.py
```

### Recursos de GPU:

- **Mínimo:** 8GB VRAM
- **Recomendado:** 12GB+ VRAM
- CPU funciona mas é significativamente mais lento

## Modo Offline

O sistema suporta modo offline para todos os modelos:

```python
# Em src/config.py
OFFLINE_MODE = True
```

**Nota:** Execute `python scripts/download_models.py` primeiro para baixar todos os modelos necessários.

## Limitações e Considerações

### Limitações do SenseVoice:

1. **Idioma:** Melhor desempenho em inglês e chinês
2. **Segmentos curtos:** Pode ter dificuldade em segmentos < 0.5s
3. **Ruído de fundo:** Ambientes ruidosos afetam precisão

### Limitações do Qwen3-TTS:

1. **Instruções complexas:** Quanto mais específica a instrução, melhor o resultado
2. **Idiomas:** Melhor qualidade em chinês, inglês e português
3. **Consistência:** Pequenas variações nas instruções podem gerar resultados diferentes

### Performance:

- Análise de emoções adiciona ~30% ao tempo de processamento
- Processamento em batch otimiza o uso de GPU
- Cache de modelos acelera execuções subsequentes

## Troubleshooting

### Problema: "SenseVoice não encontrado"
**Solução:** Verificar se o modelo está instalado:
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('FunAudioLLM/SenseVoiceSmall')"
```

### Problema: "Emoções sempre neutras"
**Solução:** 
- Verificar qualidade do áudio (deve ter fala clara)
- Segmentos muito curtos podem não ter emoção detectável
- Áudio com muito ruído afeta detecção

### Problema: "TTS não aplica emoções"
**Solução:**
- Verificar se `ENABLE_EMOTION_ANALYSIS = True` em config.py
- Confirmar que motor_tts="qwen3" (MMS não suporta instruções emocionais)
- Verificar logs para confirmar que instruções estão sendo geradas

## Referências

- **SenseVoice:** https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- **Qwen3-TTS:** https://github.com/maxwellamaral/Qwen3-TTS
- **Whisper:** https://github.com/openai/whisper
- **Documentação Transformers:** https://huggingface.co/docs/transformers

## Contribuindo

Para adicionar novas emoções ou melhorar as instruções:

1. Editar `EMOTION_INSTRUCTIONS` em `src/services/emotion.py`
2. Testar com vários segmentos de áudio
3. Ajustar instruções baseado nos resultados do Qwen3-TTS

---

**Última Atualização:** 28/01/2026
