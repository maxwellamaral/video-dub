
import os
import torch
import subprocess
from transformers import pipeline
from src.config import DEVICE
from src.utils import obter_ffmpeg_exe

FFMPEG_EXE = obter_ffmpeg_exe()

def extrair_referencia_voz(caminho_video, caminho_saida, duracao=10):
    """Extrai os primeiros segundos para referência de voz."""
    print(f"🎙️ Extraindo referência de voz ({duracao}s)...")
    video = None
    try:
        from moviepy import VideoFileClip
        video = VideoFileClip(caminho_video)
        trecho = video.subclipped(0, min(duracao, video.duration))
        trecho.audio.write_audiofile(caminho_saida, fps=22050, nbytes=2, codec='pcm_s16le', logger=None)
        print(f"✓ Referência salva em: {caminho_saida}")
        return True
    except Exception as e:
        print(f"⚠️ Erro ao extrair referência de voz: {e}")
        return False
    finally:
        if video:
            try: video.close()
            except: pass

def extrair_audio(caminho_video, caminho_audio_saida):
    """Extrai faixa de áudio completa usando ffmpeg."""
    print(f"\n📹 Extraindo áudio de: {caminho_video}")
    try:
        cmd = [
            FFMPEG_EXE, "-i", caminho_video,
            "-q:a", "9", "-n",
            caminho_audio_saida
        ]
        # output silenciado para limpeza, exceto erros
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Áudio extraído: {caminho_audio_saida}")
        return True
    except Exception as e:
        print(f"✗ Erro ao extrair áudio: {e}")
        return False

def transcrever_audio_whisper(caminho_audio, modelo="openai/whisper-base"):
    """
    Transcreve e segmenta o áudio usando Whisper (Hugging Face Pipeline).
    Retorna lista de segmentos otimizada.
    """
    print(f"\n🎙️  Transcrevendo áudio com Whisper ({modelo})...")
    
    try:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=modelo,
            device=0 if DEVICE == "cuda:0" else -1,
            torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
            chunk_length_s=30,
        )
        
        # Word-level timestamps preferencialmente
        try:
            resultado = pipe(caminho_audio, return_timestamps="word")
        except:
            print("   ⚠️ Word timestamps falhou, fallback para default.")
            resultado = pipe(caminho_audio, return_timestamps=True)
            
        return _processar_chunks_whisper(resultado)
        
    except Exception as e:
        print(f"✗ Erro na transcrição: {e}")
        return []

def _processar_chunks_whisper(resultado):
    """Reagrupa palavras/chunks em segmentos de legenda."""
    raw_chunks = resultado.get("chunks", [])
    if not raw_chunks:
        start, end = 0.0, 0.0
        # Tentar pegar do text se não houver chunks
        text = resultado.get("text", "")
        # Estimativa grosseira se não tiver time
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
        
        # Decisão de quebra
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
        
    print(f"✓ Transcrição: {len(segmentos)} segmentos gerados.")
    return segmentos
