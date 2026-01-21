
import os
import sys

# Adicionar diretório raiz ao path para imports funcionarem
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.pipeline import executar_pipeline

def menu():
    print("\n" + "="*50)
    print("   DUBBLER PRO (MODULAR v2.0)")
    print("="*50)
    print("1. MMS-TTS (Rápido, Offline)")
    print("2. Coqui XTTS (Clonagem de Voz, Qualidade)")
    
    escolha = input("\nEscolha o motor (1 ou 2): ").strip()
    motor = "coqui" if escolha == "2" else "mms"
    
    print("\nModo de Encoding:")
    print("1. Rápido (GPU NVENC) - Recomendado")
    print("2. Qualidade (CPU libx264) - Lento")
    
    enc_opt = input("Escolha (1 ou 2): ").strip()
    encoding = "qualidade" if enc_opt == "2" else "rapido"
    
    if not os.path.exists(VIDEO_ENTRADA):
        print(f"Erro: {VIDEO_ENTRADA} não encontrado.")
        return
        
    sucesso = executar_pipeline(
        caminho_video=VIDEO_ENTRADA,
        idioma_origem=IDIOMA_ORIGEM,
        idioma_destino=IDIOMA_DESTINO,
        idioma_voz="por",
        motor_tts=motor,
        modo_encoding=encoding
    )
    
    if sucesso:
        print("\n✅ Processo concluído com sucesso!")
    else:
        print("\n❌ Falha no processo.")

if __name__ == "__main__":
    menu()
