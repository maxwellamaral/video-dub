"""
Teste de Qwen3-TTS em modo OFFLINE puro.
Simula desconexão forçando variáveis de ambiente offline.
"""

import os
import sys

# FORÇAR MODO OFFLINE antes de qualquer import
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("TESTE QWEN3-TTS - MODO OFFLINE PURO")
print("="*60)
print("\n🔌 Variáveis de ambiente:")
print(f"   HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE')}")
print(f"   TRANSFORMERS_OFFLINE = {os.environ.get('TRANSFORMERS_OFFLINE')}")

print("\n📦 Tentando importar e usar Qwen3-TTS offline...")

try:
    import torch
    from qwen_tts import Qwen3TTSModel
    
    print("✅ Importação bem-sucedida!")
    
    print("\n🔄 Carregando modelo (deve usar apenas cache local)...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True  # FORÇAR apenas arquivos locais
    )
    
    print("✅ Modelo carregado do cache local!")
    
    print("\n🎤 Sintetizando áudio offline...")
    wavs, sr = model.generate_custom_voice(
        text="Testando Qwen3-TTS em modo offline completo.",
        language="Portuguese",
        speaker="Vivian"
    )
    
    print(f"✅ Síntese offline bem-sucedida!")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duração: {len(wavs[0]) / sr:.2f} segundos")
    
    # Salvar teste
    import soundfile as sf
    sf.write("output/qwen3_offline_test.wav", wavs[0], sr)
    print(f"\n💾 Áudio salvo: output/qwen3_offline_test.wav")
    
    print("\n" + "="*60)
    print("✅ SUCESSO - QWEN3-TTS FUNCIONA OFFLINE!")
    print("="*60)
    print("\n✨ O modelo está completamente funcional sem internet.")
    print("📌 Você pode reativar OFFLINE_MODE = True em config.py")
    
except Exception as e:
    print(f"\n❌ ERRO no modo offline: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("⚠️ MODO OFFLINE NÃO FUNCIONAL")
    print("="*60)
    print("\n💡 Possíveis causas:")
    print("   1. Modelos não estão completamente em cache")
    print("   2. Falta algum componente do modelo")
    print("   3. qwen-tts requer conexão para alguma operação")
    
    sys.exit(1)
