"""
Script para testar o pipeline completo com Qwen3-TTS.
Executa dublagem de vídeo usando o novo motor.
"""

import sys
import os

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import executar_pipeline

def test_pipeline_qwen3():
    """Testa pipeline completo com Qwen3-TTS."""
    print("="*60)
    print("TESTE PIPELINE COMPLETO - QWEN3-TTS")
    print("="*60)
    
    video_path = "input/video_entrada.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Vídeo não encontrado: {video_path}")
        return False
    
    print(f"\n📹 Vídeo de entrada: {video_path}")
    print(f"   Tamanho: {os.path.getsize(video_path) / (1024*1024):.2f} MB\n")
    
    resultado = executar_pipeline(
        caminho_video=video_path,
        idioma_origem="eng_Latn",
        idioma_destino="por_Latn",
        idioma_voz="por",
        motor_tts="qwen3",
        modo_encoding="rapido",
        progress_callback=print
    )
    
    print("\n" + "="*60)
    if resultado:
        print("✅ PIPELINE CONCLUÍDA COM SUCESSO!")
        print("="*60)
        print("\n📁 Vídeo dublado gerado: output/video_dublado_qwen3.mp4")
        print("📄 Legendas: output/legenda_final_sincronizada.srt")
        
        output_video = "output/video_dublado_qwen3.mp4"
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024*1024)
            print(f"📊 Tamanho do vídeo final: {size_mb:.2f} MB")
    else:
        print("❌ PIPELINE FALHOU")
        print("="*60)
    
    return resultado

if __name__ == "__main__":
    success = test_pipeline_qwen3()
    sys.exit(0 if success else 1)
