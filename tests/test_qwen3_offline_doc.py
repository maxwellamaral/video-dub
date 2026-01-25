"""
Teste Qwen3-TTS offline baseado na documentação oficial.
Usando local_files_only conforme padrão HuggingFace.
"""

import os
import sys

# NÃO forçar HF_HUB_OFFLINE (isso causa erro)
# Apenas usar local_files_only no from_pretrained

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("TESTE QWEN3-TTS - MODO OFFLINE (Método Documentação)")
print("="*60)

print("\n📚 Baseado na documentação oficial:")
print("   https://github.com/QwenLM/Qwen3-TTS")
print("\n🔧 Método: local_files_only=True (sem env vars)")

try:
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    
    print("\n📦 Carregando modelo com local_files_only=True...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True,  # Força uso apenas de arquivos locais
        trust_remote_code=True   # Necessário para modelos custom
    )
    
    print("✅ Modelo carregado do cache local!")
    
    print("\n🎤 Testando síntese...")
    wavs, sr = model.generate_custom_voice(
        text="Teste offline com configuração da documentação.",
        language="Portuguese",
        speaker="Vivian"
    )
    
    if wavs and len(wavs) > 0:
        sf.write("output/qwen3_offline_doc_method.wav", wavs[0], sr)
        print(f"✅ Síntese offline bem-sucedida!")
        print(f"   Arquivo: output/qwen3_offline_doc_method.wav")
        print(f"   Sample rate: {sr} Hz")
        
        print("\n" + "="*60)
        print("✅ MODO OFFLINE FUNCIONAL!")
        print("="*60)
        print("\n💡 Solução:")
        print("   - Usar local_files_only=True")
        print("   - Não definir HF_HUB_OFFLINE env var")
        print("   - Adicionar trust_remote_code=True")
    else:
        print("❌ Síntese retornou vazio")
        sys.exit(1)
    
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("⚠️ MODO OFFLINE AINDA NÃO FUNCIONA")
    print("="*60)
    print("\n📌 Causa provável:")
    print("   O pacote qwen-tts pode estar fazendo")
    print("   requisições de rede mesmo com local_files_only")
    
    sys.exit(1)
