
import os
import shutil
import time
from src.config import *
from src.services.audio import extrair_referencia_voz, extrair_audio, transcrever_audio_whisper
from src.services.translation import traduzir_segmentos
from src.services.tts import TTSEngine
from src.services.video import VideoEditor
from src.utils import segmentos_para_srt

def executar_pipeline(caminho_video, idioma_origem, idioma_destino, idioma_voz, motor_tts, modo_encoding):
    print("="*60)
    print(f"PIPELINE REFACTORED: {motor_tts.upper()} | {modo_encoding.upper()}")
    print("="*60)
    
    # 0. Limpeza prévia
    if os.path.exists(VIDEO_SAIDA_BASE + f"_{motor_tts}.mp4"):
        try: os.remove(VIDEO_SAIDA_BASE + f"_{motor_tts}.mp4")
        except: pass

    # 1. Extração de Áudio e Referência
    if not extrair_audio(caminho_video, AUDIO_EXTRAIDO): return False
    
    if motor_tts == "coqui":
        extrair_referencia_voz(caminho_video, AUDIO_REFERENCIA)
        
    # 2. Transcrição
    segmentos = transcrever_audio_whisper(AUDIO_EXTRAIDO)
    if not segmentos: return False
    
    # 3. Tradução
    seg_traduzidos = traduzir_segmentos(segmentos, idioma_origem, idioma_destino)
    
    # 4. Síntese TTS
    tts = TTSEngine(motor=motor_tts, idioma=idioma_voz, ref_wav=AUDIO_REFERENCIA)
    textos = [s["text"] for s in seg_traduzidos]
    # Retorna lista de (audio_np, sample_rate)
    audios = tts.sintetizar_batch(textos)
    
    # 5. Edição de Vídeo
    editor = VideoEditor(caminho_video)
    nome_saida = f"{VIDEO_SAIDA_BASE}_{motor_tts}.mp4"
    temp_files = []
    
    try:
        clips, temp_wavs, legendas_sync = editor.processar_segmentos(seg_traduzidos, audios)
        temp_files.extend(temp_wavs)
        
        ok = editor.renderizar_video(clips, nome_saida, modo=modo_encoding)
        if ok:
            # Salvar SRT final
            with open(LEGENDA_FINAL, "w", encoding="utf-8") as f:
                f.write(segmentos_para_srt(legendas_sync))
            print(f"✓ Pipeline concluída: {nome_saida}")
            
    except Exception as e:
        print(f"✗ Falha na edição: {e}")
        ok = False
        
    finally:
        editor.close()
        # Limpeza robusta
        print("   🧹 Limpando temporários...")
        for f in temp_files:
            try:
                if os.path.exists(f): os.remove(f)
            except: 
                # Retry logic simples
                time.sleep(0.5)
                try: os.remove(f)
                except: pass
                
    return ok

if __name__ == "__main__":
    # Teste rápido se rodado direto
    executar_pipeline(VIDEO_ENTRADA, IDIOMA_ORIGEM, IDIOMA_DESTINO, "por", "mms", "rapido")
