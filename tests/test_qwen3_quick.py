"""
Script de teste rápido para verificar a funcionalidade do Qwen3-TTS.

Execute apenas para verificar se o modelo carrega e sintetiza áudio corretamente.
"""

import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import soundfile as sf
from src.services.tts import TTSEngine

def test_qwen3_synthesis():
    """Testa síntese de áudio com Qwen3-TTS."""
    print("="*60)
    print("TESTE RÁPIDO QWEN3-TTS")
    print("="*60)
    
    print("\n📝 Inicializando TTSEngine com Qwen3...")
    tts = TTSEngine(
        motor="qwen3",
        idioma="por",
        log_callback=print
    )
    
    textos = [
        "Olá! Este é um teste de síntese de voz com Qwen3-TTS.",
        "O modelo suporta múltiplos idiomas com alta qualidade.",
        "A latência é extremamente baixa, ideal para aplicações em tempo real."
    ]
    
    print(f"\n🎤 Sintetizando {len(textos)} frases...")
    audios = tts.sintetizar_batch(textos)
    
    # Salvar áudios
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (audio, sr) in enumerate(audios):
        if audio is not None:
            filename = f"{output_dir}/qwen3_test_{i+1}.wav"
            sf.write(filename, audio, sr)
            print(f"   ✓ Áudio {i+1} salvo: {filename}")
        else:
            print(f"   ✗ Áudio {i+1} falhou")
    
    print("\n" + "="*60)
    print("✅ TESTE CONCLUÍDO!")
    print("="*60)
    print(f"\nAudios salvos em: {os.path.abspath(output_dir)}/")
    print("Reproduza os arquivos .wav para verificar a qualidade.")

if __name__ == "__main__":
    test_qwen3_synthesis()
