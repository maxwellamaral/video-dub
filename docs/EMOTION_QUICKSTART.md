# 🎭 Análise de Emoções - Guia Rápido

## Novidade: Dublagem com Expressividade Emocional

O sistema agora detecta automaticamente as **emoções** no áudio original e as utiliza para gerar uma dublagem mais **expressiva e natural**!

## Como Funciona

### 1️⃣ Transcrição + Análise de Emoções
```
Áudio Original → Whisper (texto) + SenseVoice (emoções)
```

### 2️⃣ Tradução Preservando Emoções
```
Texto em inglês (feliz) → Texto em português (feliz)
```

### 3️⃣ Síntese TTS Expressiva
```
Qwen3-TTS recebe:
- Texto: "Olá, como você está?"
- Instrução: "Fale com tom alegre e entusiasmado..."

Resultado: Áudio dublado com emoção apropriada! 🎉
```

## Emoções Suportadas

| Emoção | Tag | Exemplo de Instrução TTS |
|--------|-----|--------------------------|
| 😊 Happy | `[FELIZ]` | "Fale com tom alegre e entusiasmado..." |
| 😢 Sad | `[TRISTE]` | "Fale com tom triste e melancólico..." |
| 😠 Angry | `[ZANGADO]` | "Fale com tom zangado, voz elevada..." |
| 😐 Neutral | (sem tag) | "Fale com tom neutro e equilibrado..." |
| 😨 Fearful | `[AMEDRONTADO]` | "Fale com tom hesitante e trêmulo..." |
| 🤢 Disgusted | `[ENOJADO]` | "Fale com tom de desgosto..." |
| 😮 Surprised | `[SURPRESO]` | "Fale com tom surpreso e espantado..." |

## Exemplo de Uso

### Configuração (src/config.py)
```python
# Habilitar análise de emoções
ENABLE_EMOTION_ANALYSIS = True  # ✅ Ativado

# Incluir tags nas legendas
INCLUDE_EMOTION_TAGS_IN_SUBTITLES = True  # [FELIZ], [TRISTE], etc.
```

### Executar Pipeline
```python
from src.pipeline import executar_pipeline

executar_pipeline(
    caminho_video="input/video.mp4",
    idioma_origem="eng_Latn",
    idioma_destino="por_Latn",
    idioma_voz="por",
    motor_tts="qwen3",  # ⚠️ Requer Qwen3 para emoções
    modo_encoding="qualidade",
    qwen3_mode="custom",
    qwen3_speaker="vivian"
)
```

### Resultado

**Legendas geradas:**
```srt
1
00:00:01,000 --> 00:00:03,500
[FELIZ] Olá, como você está?

2
00:00:04,000 --> 00:00:07,200
[TRISTE] Estou muito cansado hoje...

3
00:00:08,000 --> 00:00:11,500
Este é um diálogo neutro sem emoção.
```

**Áudio dublado:** Cada fala é sintetizada com a emoção detectada! 🎙️

## Estatísticas de Emoções

Durante o processamento, você verá estatísticas das emoções detectadas:

```
📊 Estatísticas de Emoções:
   Total de segmentos: 45
   Emoção predominante: happy
   - happy: 18 (40.0%)
   - neutral: 15 (33.3%)
   - sad: 8 (17.8%)
   - angry: 4 (8.9%)
```

## Arquivos Gerados

```
output/
├── legenda_original.srt          # Com emoções detectadas
├── legenda_traduzida.srt         # Tradução + emoções
├── legenda_final_sincronizada.srt # Final com emoções
└── video_dublado_qwen3.mp4       # Vídeo com áudio expressivo
```

## Desabilitar Emoções

Para voltar ao pipeline original (sem análise de emoções):

```python
# Em src/config.py
ENABLE_EMOTION_ANALYSIS = False
```

## Modelos Utilizados

- **Transcrição:** OpenAI Whisper
- **Emoções:** FunAudioLLM SenseVoiceSmall (~2GB)
- **TTS:** Qwen3-TTS CustomVoice (~3.4GB)

## Requisitos

- Python 3.11+
- GPU com 8GB+ VRAM (recomendado)
- Modelos baixados (execute `python scripts/download_models.py`)

## Documentação Completa

Para detalhes técnicos completos, consulte:
📖 [docs/EMOTION_ANALYSIS.md](docs/EMOTION_ANALYSIS.md)

## Testes

```bash
# Executar testes de emoções
pytest tests/test_emotion_pipeline.py -v

# Teste completo de integração
pytest tests/test_emotion_pipeline.py::TestPipelineComEmocoes -v
```

## Limitações

- **Idiomas:** Melhor desempenho em inglês e chinês
- **Áudio limpo:** Ruído de fundo reduz precisão
- **Segmentos curtos:** < 0.5s podem não ter emoção detectável
- **TTS:** Apenas Qwen3-TTS suporta instruções emocionais

## Troubleshooting

### Emoções sempre neutras?
- Verifique qualidade do áudio (fala clara, pouco ruído)
- Segmentos muito curtos podem não ter emoção detectável
- Confirme que `ENABLE_EMOTION_ANALYSIS = True`

### TTS não aplica emoções?
- Use `motor_tts="qwen3"` (MMS não suporta instruções)
- Verifique logs para confirmar geração de instruções
- Teste com áudio com emoções evidentes

## Contribuindo

Ajude a melhorar as instruções emocionais editando:
`src/services/emotion.py` → `EMOTION_INSTRUCTIONS`

Teste suas mudanças e compartilhe os resultados!

---

**Desenvolvido com ❤️ para dublagens mais naturais e expressivas**
