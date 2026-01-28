
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


def segmentos_para_srt_com_emocao(segmentos, incluir_tag_emocao=True):
    """
    Converte segmentos com emoções para formato SRT.
    
    Gera legendas no formato SRT incluindo opcionalmente tags de emoção
    no texto. As emoções são inseridas como [EMOÇÃO] antes do texto.
    
    Exemplo de saída:
        1
        00:00:01,000 --> 00:00:05,500
        [FELIZ] Olá, como você está?
        
        2
        00:00:06,000 --> 00:00:10,200
        [TRISTE] Estou muito cansado hoje...
    
    Args:
        segmentos (list): Lista de dicts com campos obrigatórios:
                         - 'start': float (tempo inicial em segundos)
                         - 'end': float (tempo final em segundos)
                         - 'text': str (texto da legenda)
                         E campos opcionais:
                         - 'emotion': str (código da emoção, ex: 'happy')
                         - 'emotion_pt': str (emoção em português, ex: 'feliz')
        incluir_tag_emocao (bool): Se True, inclui tag [EMOÇÃO] no texto.
                                   Se False, gera SRT padrão sem tags.
    
    Returns:
        str: Conteúdo formatado em SRT.
    """
    linhas = []
    
    for i, seg in enumerate(segmentos, 1):
        inicio = formatar_tempo_srt(seg["start"])
        fim = formatar_tempo_srt(seg["end"])
        texto = seg["text"].strip()
        
        if not texto:
            continue
        
        # Adicionar tag de emoção se disponível e solicitado
        if incluir_tag_emocao and "emotion_pt" in seg:
            emotion = seg.get("emotion", "neutral")
            # Não incluir tag se for neutro (evita poluição visual)
            if emotion != "neutral":
                emotion_tag = f"[{seg['emotion_pt'].upper()}]"
                texto = f"{emotion_tag} {texto}"
        
        linhas.append(f"{i}")
        linhas.append(f"{inicio} --> {fim}")
        linhas.append(texto)
        linhas.append("")
    
    return "\n".join(linhas)


def extrair_estatisticas_emocoes(segmentos):
    """
    Extrai estatísticas sobre as emoções detectadas nos segmentos.
    
    Útil para análise e visualização da distribuição emocional no vídeo.
    
    Args:
        segmentos (list): Lista de segmentos com campo 'emotion'.
    
    Returns:
        dict: Dicionário com estatísticas:
              - 'total': número total de segmentos
              - 'emocoes': dict com contagem por emoção
              - 'predominante': emoção mais frequente
              - 'distribuicao_percentual': dict com % por emoção
    """
    from collections import Counter
    
    total = len(segmentos)
    if total == 0:
        return {
            "total": 0,
            "emocoes": {},
            "predominante": None,
            "distribuicao_percentual": {}
        }
    
    # Contar emoções
    emocoes = [seg.get("emotion", "neutral") for seg in segmentos]
    contagem = Counter(emocoes)
    
    # Calcular percentuais
    distribuicao = {
        emocao: (count / total) * 100
        for emocao, count in contagem.items()
    }
    
    # Identificar emoção predominante
    predominante = contagem.most_common(1)[0][0] if contagem else None
    
    return {
        "total": total,
        "emocoes": dict(contagem),
        "predominante": predominante,
        "distribuicao_percentual": distribuicao
    }
