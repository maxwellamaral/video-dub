
import os
import shutil
import time
from src.config import *
from src.services.audio import extrair_referencia_voz, extrair_audio, transcrever_audio_whisper
from src.services.translation import traduzir_segmentos
from src.services.tts import TTSEngine
from src.services.video import VideoEditor
from src.utils import segmentos_para_srt

def executar_pipeline(caminho_video, idioma_origem, idioma_destino, idioma_voz, 
                     motor_tts, modo_encoding, progress_callback=None,
                     qwen3_mode="custom", qwen3_speaker="vivian", qwen3_instruct=""):
    """
    Pipeline principal de dublagem de vídeo.

    Args:
        caminho_video (str): Caminho do vídeo de entrada.
        idioma_origem (str): Código do idioma original (ex: 'eng_Latn').
        idioma_destino (str): Código do idioma de destino (ex: 'por_Latn').
        idioma_voz (str): Código do idioma da voz gerada (ex: 'por').
        motor_tts (str): Motor de TTS a usar ('mms', 'coqui', 'qwen3').
        modo_encoding (str): Modo de codificação ('rapido' ou 'qualidade').
        progress_callback (callable, optional): Função para notificar progresso.
        qwen3_mode (str): Modalidade Qwen3 ('custom', 'design', 'clone').
        qwen3_speaker (str): Speaker para CustomVoice (ex: 'Vivian').
        qwen3_instruct (str): Instrução de voz para CustomVoice/VoiceDesign.

    Returns:
        bool: True se o pipeline foi executado com sucesso, False caso contrário.
    """
    def log(msg):
        print(msg)
        if progress_callback:
            try:
                progress_callback(msg)
            except: pass

    log("="*60)
    log(f"PIPELINE WEB: {motor_tts.upper()} | {modo_encoding.upper()}")
    log("="*60)
    
    # 0. Limpeza prévia
    nome_saida = f"{VIDEO_SAIDA_BASE}_{motor_tts}.mp4"
    if os.path.exists(nome_saida):
        try: os.remove(nome_saida)
        except: pass
    
    # Limpeza de arquivos de legenda e áudio antigos
    for arquivo in [AUDIO_EXTRAIDO, LEGENDA_ORIGINAL, LEGENDA_TRADUZIDA, LEGENDA_FINAL]:
        if os.path.exists(arquivo):
            try: os.remove(arquivo)
            except: pass

    # 1. Extração de Áudio e Referência
    log("1. Extraindo áudio original...")
    if not extrair_audio(caminho_video, AUDIO_EXTRAIDO, log_callback=log): 
        log("❌ Falha na extração de áudio.")
        return False
    
    if motor_tts == "coqui":
        log("1.1. Extraindo referência de voz (Clonagem)...")
        extrair_referencia_voz(caminho_video, AUDIO_REFERENCIA, log_callback=log)
        
    # 2. Transcrição
    log("2. Transcrevendo áudio (Whisper)...")
    segmentos = transcrever_audio_whisper(AUDIO_EXTRAIDO, log_callback=log)
    if not segmentos: 
        log("❌ Nenhum diálogo detectado ou falha na transcrição.")
        return False
    
    # Salvar legenda original
    with open(LEGENDA_ORIGINAL, "w", encoding="utf-8") as f:
        f.write(segmentos_para_srt(segmentos))
    
    # 3. Tradução
    log(f"3. Traduzindo para {idioma_destino} (NLLB)...")
    seg_traduzidos = traduzir_segmentos(segmentos, idioma_origem, idioma_destino, log_callback=log)
    
    # Salvar legenda traduzida
    with open(LEGENDA_TRADUZIDA, "w", encoding="utf-8") as f:
        f.write(segmentos_para_srt(seg_traduzidos))
    
    # 4. Síntese TTS
    log(f"4. Sintetizando Voz ({motor_tts})...")
    tts = TTSEngine(
        motor=motor_tts, 
        idioma=idioma_voz, 
        ref_wav=AUDIO_REFERENCIA,
        log_callback=log,
        qwen3_mode=qwen3_mode,
        qwen3_speaker=qwen3_speaker,
        qwen3_instruct=qwen3_instruct
    )
    
    textos = [s["text"] for s in seg_traduzidos]
    
    # Retorna lista de (audio_np, sample_rate)
    log(f"   Gerando áudio para {len(textos)} segmentos...")
    audios = tts.sintetizar_batch(textos)
    
    # 5. Edição de Vídeo
    log("5. Editando e Sincronizando Vídeo...")
    editor = VideoEditor(caminho_video)
    temp_files = []
    ok = False
    
    try:
        clips, temp_wavs, legendas_sync = editor.processar_segmentos(seg_traduzidos, audios, log_callback=log)
        temp_files.extend(temp_wavs)
        
        log(f"   Renderizando vídeo final: {os.path.basename(nome_saida)}")
        ok = editor.renderizar_video(clips, nome_saida, modo=modo_encoding, log_callback=log)
        if ok:
            # Salvar SRT final
            with open(LEGENDA_FINAL, "w", encoding="utf-8") as f:
                f.write(segmentos_para_srt(legendas_sync))
            log(f"✅ Pipeline concluída com sucesso!")
            
    except Exception as e:
        log(f"❌ Falha na edição: {e}")
        ok = False
        import traceback
        traceback.print_exc()
        
    finally:
        editor.close()
        # Limpeza robusta
        log("   🧹 Limpando arquivos temporários...")
        for f in temp_files:
            try:
                if os.path.exists(f): os.remove(f)
            except: 
                time.sleep(0.5)
                try: os.remove(f)
                except: pass
                
    return ok
