
import os
import torch
import subprocess
from transformers import pipeline
from src.config import DEVICE
from src.utils import obter_ffmpeg_exe

FFMPEG_EXE = obter_ffmpeg_exe()

def extrair_referencia_voz(caminho_video, caminho_saida, duracao=10, log_callback=None):
    """
    Extrai os primeiros segundos de áudio do vídeo para usar como referência de clonagem.
    
    Usado principalmente pelo Coqui TTS para capturar o timbre original da voz.

    Args:
        caminho_video (str): Path do vídeo de entrada.
        caminho_saida (str): Path onde o áudio de referência será salvo (.wav).
        duracao (int, optional): Duração em segundos do trecho a extrair. Default: 10s.
        log_callback (callable, optional): Função para logar mensagens.

    Returns:
        bool: True se sucesso, False caso contrário.
    """
    msg = f"🎙️ Extraindo referência de voz ({duracao}s)..."
    if log_callback: log_callback(msg)
    else: print(msg)

    video = None
    try:
        from moviepy import VideoFileClip
        video = VideoFileClip(caminho_video)
        trecho = video.subclipped(0, min(duracao, video.duration))
        trecho.audio.write_audiofile(caminho_saida, fps=22050, nbytes=2, codec='pcm_s16le', logger=None)
        
        msg_ok = f"✓ Referência salva em: {caminho_saida}"
        if log_callback: log_callback(msg_ok)
        else: print(msg_ok)
        
        return True
    except Exception as e:
        msg_err = f"⚠️ Erro ao extrair referência de voz: {e}"
        if log_callback: log_callback(msg_err)
        else: print(msg_err)
        return False
    finally:
        if video:
            try: video.close()
            except: pass

def extrair_audio(caminho_video, caminho_audio_saida, log_callback=None):
    """
    Extrai a faixa de áudio completa de um vídeo usando FFmpeg.
    
    O áudio é extraído sem recompressão desnecessária (-q:a 9) para garantir qualidade
    na transcrição subsequente.

    Args:
        caminho_video (str): Path do vídeo de entrada.
        caminho_audio_saida (str): Path de saída do áudio (.wav).
        log_callback (callable, optional): Função para logar mensagens.

    Returns:
        bool: True se sucesso, False caso contrário.
    """
    msg = f"\n📹 Extraindo áudio de: {caminho_video}"
    if log_callback: log_callback(msg)
    else: print(msg)

    try:
        cmd = [
            FFMPEG_EXE, "-i", caminho_video,
            "-q:a", "9", "-n",
            caminho_audio_saida
        ]
        # output silenciado para limpeza, exceto erros
        subprocess.run(cmd, check=True, capture_output=True)
        
        msg_ok = f"✓ Áudio extraído: {caminho_audio_saida}"
        if log_callback: log_callback(msg_ok)
        else: print(msg_ok)
        
        return True
    except Exception as e:
        msg_err = f"✗ Erro ao extrair áudio: {e}"
        if log_callback: log_callback(msg_err)
        else: print(msg_err)
        return False

def transcrever_audio_whisper(caminho_audio, modelo="openai/whisper-base", log_callback=None):
    """
    Transcreve áudio para texto com timestamps precisos usando o modelo Whisper.

    Utiliza a pipeline `automatic-speech-recognition` da Hugging Face.
    Tenta priorizar timestamps em nível de palavra (word-level) para melhor
    sincronização labial/segmentação.

    Args:
        caminho_audio (str): Path do arquivo de áudio (.wav).
        modelo (str, optional): ID do modelo Whisper no Hugging Face. Default: "openai/whisper-base".
        log_callback (callable, optional): Função para logar mensagens.

    Returns:
        list: Lista de dicionários de segmentos processados (ver `_processar_chunks_whisper`).
    """
    msg = f"\n🎙️  Transcrevendo áudio com Whisper ({modelo})..."
    if log_callback: log_callback(msg)
    else: print(msg)
    
    try:
        if log_callback: log_callback("   Carregando modelo Whisper...")
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=modelo,
            device=0 if DEVICE == "cuda:0" else -1,
            torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
            chunk_length_s=30,
        )
        
        # Word-level timestamps preferencialmente
        try:
            if log_callback: log_callback("   Processando (word timestamps)...")
            resultado = pipe(caminho_audio, return_timestamps="word")
        except:
            warn = "   ⚠️ Word timestamps falhou, fallback para default."
            if log_callback: log_callback(warn)
            else: print(warn)
            resultado = pipe(caminho_audio, return_timestamps=True)
            
        return _processar_chunks_whisper(resultado, log_callback)
        
    except Exception as e:
        err = f"✗ Erro na transcrição: {e}"
        if log_callback: log_callback(err)
        else: print(err)
        return []

def _processar_chunks_whisper(resultado, log_callback=None):
    """Reagrupa palavras/chunks em segmentos de legenda."""
    raw_chunks = resultado.get("chunks", [])
    if not raw_chunks:
        text = resultado.get("text", "")
        return [{"start": 0.0, "end": 5.0, "text": text}] if text else []

    segmentos = []
    MAX_CHARS = 80
    MAX_DUR = 7.0
    MIN_PAUSE = 0.5
    
    buffer_words = []
    seg_start = 0.0
    last_end = 0.0
    buffer_len = 0
    
    for chunk in raw_chunks:
        text = chunk.get("text", "").strip()
        times = chunk.get("timestamp")
        
        if not text: continue
        
        if isinstance(times, (list, tuple)):
            start, end = times
        else:
            start, end = last_end, last_end + 1.0 # fallback
            
        if start is None: start = last_end
        if end is None: end = start + 0.3
        
        # Início do primeiro buffer
        if not buffer_words and not segmentos:
            seg_start = start
            if buffer_words == []: seg_start = start # Reset para novo segmento
            
        pause = start - last_end
        duration = end - seg_start
        
        should_break = False
        if buffer_words:
            if pause > MIN_PAUSE: should_break = True
            elif duration > MAX_DUR: should_break = True
            elif buffer_len + len(text) > MAX_CHARS: should_break = True
            elif buffer_words[-1][-1] in ".?!": should_break = True
            
        if should_break:
            segmentos.append({
                "start": seg_start,
                "end": last_end,
                "text": " ".join(buffer_words)
            })
            buffer_words = []
            seg_start = start
            buffer_len = 0
            
        buffer_words.append(text)
        buffer_len += len(text) + 1
        last_end = end
        
    if buffer_words:
        segmentos.append({
            "start": seg_start,
            "end": last_end,
            "text": " ".join(buffer_words)
        })
        
    msg = f"✓ Transcrição: {len(segmentos)} segmentos gerados."
    if log_callback: log_callback(msg)
    else: print(msg)
    
    return segmentos
