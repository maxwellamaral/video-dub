"""
Script para baixar todos os modelos necessários para execução offline.

Execute este script uma vez com conexão à internet para baixar todos os modelos.
Depois, o projeto funcionará sem conexão.
"""

import os
from transformers import pipeline, VitsModel, AutoTokenizer
from TTS.api import TTS
import torch

def download_models():
    print("="*60)
    print("BAIXANDO MODELOS PARA EXECUÇÃO OFFLINE")
    print("="*60)
    
    # 1. Whisper (Transcrição de Áudio)
    print("\n1️⃣ Baixando Whisper (ASR)...")
    try:
        pipe_asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=-1  # CPU para download
        )
        print("   ✓ Whisper baixado com sucesso!")
    except Exception as e:
        print(f"   ✗ Erro ao baixar Whisper: {e}")
    
    # 2. NLLB (Tradução)
    print("\n2️⃣ Baixando NLLB-200 (Tradução)...")
    try:
        pipe_translation = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            device=-1
        )
        print("   ✓ NLLB-200 baixado com sucesso!")
    except Exception as e:
        print(f"   ✗ Erro ao baixar NLLB: {e}")
    
    # 3. MMS-TTS (Síntese de Voz)
    print("\n3️⃣ Baixando MMS-TTS (Português)...")
    try:
        modelo_nome = "facebook/mms-tts-por"
        tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
        model = VitsModel.from_pretrained(modelo_nome)
        print("   ✓ MMS-TTS (por) baixado com sucesso!")
    except Exception as e:
        print(f"   ✗ Erro ao baixar MMS-TTS: {e}")
    
    # 4. Coqui XTTS v2 (Opcional - clonagem de voz)
    print("\n4️⃣ Baixando Coqui XTTS v2 (Clonagem de Voz)...")
    try:
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        # Patch para PyTorch 2.6+
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        torch.load = original_load
        
        print("   ✓ Coqui XTTS v2 baixado com sucesso!")
    except Exception as e:
        print(f"   ✗ Erro ao baixar Coqui: {e}")
    
    # Informações sobre cache
    print("\n" + "="*60)
    print("✅ DOWNLOAD CONCLUÍDO!")
    print("="*60)
    print("\nOs modelos foram salvos no cache local:")
    
    if os.name == 'nt':  # Windows
        cache_path = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"   📁 {cache_path}")
    else:
        print("   📁 ~/.cache/huggingface/hub")
    
    print("\n💡 Agora você pode executar o projeto sem conexão à internet!")
    print("   Use: uv run python main_refactored.py")

if __name__ == "__main__":
    download_models()
