
import os
import torch

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

# Defina o dispositivo: "cuda:0" para GPU ou "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Qwen3-TTS Configuration
# ============================================================================
QWEN3_DEFAULT_SPEAKER = "vivian"  # Lowercase conforme modelo
QWEN3_MODELO_VARIANTE = "1.7B"
QWEN3_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Speakers disponíveis para Qwen3-CustomVoice (APENAS os suportados pelo modelo)
# Fonte: Modelo Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN3_SPEAKERS = {
    "vivian": "feminino, voz suave e expressiva",
    "ryan": "masculino, voz profissional e clara",
    "aiden": "masculino, voz articulada",
    "dylan": "masculino, voz jovem e casual",
    "eric": "masculino, voz madura",
    "serena": "feminino, voz serena e calma",
    "ono_anna": "feminino, voz doce",
    "sohee": "feminino, voz gentil",
    "uncle_fu": "masculino, voz mais velha"
}

# ============================================================================
# ANÁLISE DE EMOÇÕES - SenseVoiceSmall
# ============================================================================
# Modelo para detecção de emoções em áudio
# O SenseVoice detecta emoções durante a transcrição e as integra ao pipeline
SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"

# Habilitar/Desabilitar análise de emoções
# Quando True, o pipeline usa SenseVoice para detectar emoções e aplicá-las ao TTS
# Quando False, usa apenas transcrição simples com Whisper
ENABLE_EMOTION_ANALYSIS = True

# Incluir tags de emoção nas legendas
# Formato: [FELIZ] Texto da legenda
INCLUDE_EMOTION_TAGS_IN_SUBTITLES = True

# Emoções suportadas pelo SenseVoice
# Estas são detectadas automaticamente e mapeadas para instruções do Qwen3-TTS
SUPPORTED_EMOTIONS = [
    "neutral",   # neutro
    "happy",     # feliz
    "sad",       # triste
    "angry",     # zangado
    "fearful",   # amedrontado
    "disgusted", # enojado
    "surprised"  # surpreso
]

# ============================================================================
# MODO OFFLINE - Desabilita verificação de internet para modelos Hugging Face
# ============================================================================
# Quando True, força o uso de modelos apenas do cache local
# Execute 'python download_models.py' primeiro para baixar os modelos
OFFLINE_MODE = False 

# Configurar variáveis de ambiente para modo offline
# IMPORTANTE: Não usar HF_HUB_OFFLINE pois conflita com local_files_only do Qwen3-TTS
# O Qwen3-TTS usa local_files_only=True internamente no from_pretrained
if OFFLINE_MODE:
    # Removido: os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Caminhos de Diretórios
BASE_DIR = os.getcwd() # Ou definir um path fixo se preferir
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
YOUTUBE_DOWNLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Garantir existência
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(YOUTUBE_DOWNLOAD_DIR, exist_ok=True)

# Arquivos Padrão
VIDEO_ENTRADA = os.path.join(INPUT_DIR, "video_entrada.mp4")
AUDIO_EXTRAIDO = os.path.join(OUTPUT_DIR, "audio_extraido.wav")
AUDIO_REFERENCIA = os.path.join(OUTPUT_DIR, "referencia_voz.wav")
AUDIO_TRADUZIDO = os.path.join(OUTPUT_DIR, "audio_traduzido.wav")
VIDEO_SAIDA_BASE = os.path.join(OUTPUT_DIR, "video_dublado")
LEGENDA_ORIGINAL = os.path.join(OUTPUT_DIR, "legenda_original.srt")
LEGENDA_TRADUZIDA = os.path.join(OUTPUT_DIR, "legenda_traduzida.srt")
LEGENDA_FINAL = os.path.join(OUTPUT_DIR, "legenda_final_sincronizada.srt")

# Configurações de Idioma Padrão
IDIOMA_ORIGEM = "eng_Latn"      # Inglês
IDIOMA_DESTINO = "por_Latn"    # Português
IDIOMA_VOZ_PADRAO = "por"

# Opções Disponíveis
MOTORES_TTS = ["mms", "coqui", "qwen3"]
MODOS_ENCODING = ["rapido", "qualidade"]

# Configurações Qwen3-TTS
QWEN3_DEFAULT_SPEAKER = "Vivian"  # Speaker padrão para português
QWEN3_MODELO_VARIANTE = "1.7B"    # ou "0.6B" para menor uso de memória
QWEN3_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
