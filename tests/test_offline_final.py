"""
Teste final de Qwen3-TTS em modo offline puro com pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("TESTE FINAL - PIPELINE QWEN3 OFFLINE")
print("="*60)

# Verificar env vars ANTES de importar config
print("\n🔍 Variáveis de ambiente ANTES imports:")
print(f"   HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE', 'NÃO DEFINIDA')}")
print(f"   TRANSFORMERS_OFFLINE = {os.environ.get('TRANSFORMERS_OFFLINE', 'NÃO DEFINIDA')}")

# Importar config
from src.config import VIDEO_ENTRADA, IDIOMA_ORIGEM, IDIOMA_DESTINO
from src.pipeline import executar_pipeline

print("\n🔍 Variáveis de ambiente APÓS imports:")
print(f"   HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE', 'NÃO DEFINIDA')}")
print(f"   TRANSFORMERS_OFFLINE = {os.environ.get('TRANSFORMERS_OFFLINE', 'NÃO DEFINIDA')}")

print("\n▶️ Executando pipeline...")
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
    print("✅ SUCESSO - OFFLINE MODE FUNCIONAL!")
    print("="*60)
else:
    print("❌ FALHA")
    print("="*60)

sys.exit(0 if sucesso else 1)
