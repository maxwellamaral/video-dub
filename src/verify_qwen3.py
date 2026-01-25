"""
Script para verificar se os modelos Qwen3-TTS foram baixados corretamente.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("VERIFICAÇÃO DOS MODELOS QWEN3-TTS")
print("="*60)

# 1. Verificar se qwen-tts está instalado
print("\n1️⃣ Verificando instalação do pacote qwen-tts...")
try:
    import qwen_tts
    print("   ✅ Pacote qwen-tts instalado")
    print(f"   Versão: {qwen_tts.__version__ if hasattr(qwen_tts, '__version__') else 'desconhecida'}")
except ImportError as e:
    print(f"   ❌ qwen-tts não instalado: {e}")
    sys.exit(1)

# 2. Tentar carregar modelo CustomVoice
print("\n2️⃣ Tentando carregar modelo CustomVoice...")
try:
    from qwen_tts import Qwen3TTSModel
    print("   Carregando Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice...")
    
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cpu",  # Usar CPU para teste rápido
        dtype="bfloat16" if hasattr(__import__('torch'), 'bfloat16') else "float32"
    )
    print("   ✅ Modelo CustomVoice carregado com sucesso!")
    
    # Verificar speakers disponíveis
    if hasattr(model, 'get_supported_speakers'):
        speakers = model.get_supported_speakers()
        print(f"   Speakers disponíveis: {len(speakers) if speakers else 0}")
        if speakers:
            print(f"   Exemplos: {', '.join(list(speakers)[:5])}")
    
    del model  # Limpar memória
    
except Exception as e:
    print(f"   ❌ Erro ao carregar modelo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Verificar cache do HuggingFace
print("\n3️⃣ Verificando cache do HuggingFace...")
try:
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if cache_dir.exists():
        qwen_models = list(cache_dir.glob("models--Qwen--*TTS*"))
        print(f"   Modelos Qwen3-TTS encontrados: {len(qwen_models)}")
        
        for model_dir in qwen_models:
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            print(f"   ✅ {model_name} ({size_gb:.2f} GB)")
    else:
        print(f"   ⚠️ Cache dir não encontrado: {cache_dir}")
        
except Exception as e:
    print(f"   ⚠️ Erro ao verificar cache: {e}")

# 4. Teste de síntese rápida
print("\n4️⃣ Teste de síntese de áudio...")
try:
    import torch
    from qwen_tts import Qwen3TTSModel
    
    print("   Carregando modelo para teste...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cpu",
        dtype=torch.float32
    )
    
    print("   Sintetizando frase de teste...")
    wavs, sr = model.generate_custom_voice(
        text="Teste de verificação do Qwen3-TTS.",
        language="Portuguese",
        speaker="Vivian"
    )
    
    if wavs is not None and len(wavs) > 0:
        print(f"   ✅ Síntese bem-sucedida! Sample rate: {sr} Hz, Tamanho: {len(wavs[0])} samples")
    else:
        print("   ❌ Síntese retornou vazio")
        sys.exit(1)
    
    del model
    
except Exception as e:
    print(f"   ❌ Erro no teste de síntese: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ VERIFICAÇÃO COMPLETA - QWEN3-TTS FUNCIONANDO!")
print("="*60)
print("\n📌 Os modelos estão baixados e funcionando corretamente.")
print("💡 Você pode prosseguir com as melhorias planejadas.")
