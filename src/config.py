
import os
import torch

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

# Defina o dispositivo: "cuda:0" para GPU ou "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Caminhos de Diretórios
BASE_DIR = os.getcwd() # Ou definir um path fixo se preferir
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Garantir existência
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
MOTORES_TTS = ["mms", "coqui"]
MODOS_ENCODING = ["rapido", "qualidade"]
