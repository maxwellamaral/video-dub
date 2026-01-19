
import os
import shutil
import sys

def obter_ffmpeg_exe():
    """Retorna o caminho do executável ffmpeg."""
    # Tentar obter via imageio_ffmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Configurar PATH para subprocessos
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        if ffmpeg_dir not in os.environ["PATH"]:
             os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
             
        # Compatibilidade com libs que buscam 'ffmpeg.exe' exato
        ffmpeg_base = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if not os.path.exists(ffmpeg_base):
             try:
                 shutil.copy(ffmpeg_exe, ffmpeg_base)
             except: pass
             
        return ffmpeg_exe
    except:
        return "ffmpeg" # Fallback

def formatar_tempo_srt(segundos):
    """Converte segundos para formato SRT (HH:MM:SS,mmm)"""
    if segundos is None or segundos < 0:
        segundos = 0
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    secs = int(segundos % 60)
    millis = int((segundos % 1) * 1000)
    return f"{horas:02d}:{minutos:02d}:{secs:02d},{millis:03d}"

def segmentos_para_srt(segmentos):
    """
    Converte lista de segmentos para formato SRT.
    Args:
        segmentos: Lista de dicts com {'start': float, 'end': float, 'text': str}
    """
    linhas = []
    for i, seg in enumerate(segmentos, 1):
        inicio = formatar_tempo_srt(seg["start"])
        fim = formatar_tempo_srt(seg["end"])
        texto = seg["text"].strip()
        
        if texto:
            linhas.append(f"{i}")
            linhas.append(f"{inicio} --> {fim}")
            linhas.append(texto)
            linhas.append("")
    return "\n".join(linhas)

def segmentos_para_texto(segmentos):
    """Extrai apenas o texto concatenado."""
    return " ".join([s["text"] for s in segmentos if s["text"].strip()])
