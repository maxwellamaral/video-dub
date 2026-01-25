"""
Teste direto do pipeline com Qwen3-TTS sem menu interativo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import VIDEO_ENTRADA, IDIOMA_ORIGEM, IDIOMA_DESTINO
from src.pipeline import executar_pipeline

print("="*60)
print("TESTE PIPELINE - QWEN3-TTS")
print("="*60)
print(f"\n📹 Vídeo: {VIDEO_ENTRADA}")
print("🎤 Motor TTS: Qwen3-TTS")
print("⚡ Encoding: Rápido (NVENC)\n")

sucesso = executar_pipeline(
    caminho_video=VIDEO_ENTRADA,
    idioma_origem=IDIOMA_ORIGEM,
    idioma_destino=IDIOMA_DESTINO,
    idioma_voz="por",
    motor_tts="qwen3",
    modo_encoding="rapido"
)

print("\n" + "="*60)
if sucesso:
    print("✅ SUCESSO! Vídeo dublado gerado.")
    print("="*60)
    print("\n📁 Saída: output/video_dublado_qwen3.mp4")
else:
    print("❌ FALHA no pipeline")
    print("="*60)

sys.exit(0 if sucesso else 1)
