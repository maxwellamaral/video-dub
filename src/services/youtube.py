
import os
import re
from typing import Optional, Callable
import yt_dlp


def validar_url_youtube(url: str) -> bool:
    """
    Valida se a URL é de um vídeo do YouTube.
    
    Args:
        url: URL a ser validada
        
    Returns:
        True se a URL é válida, False caso contrário
    """
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    
    youtube_regex_match = re.match(youtube_regex, url)
    return youtube_regex_match is not None


def extrair_video_id(url: str) -> Optional[str]:
    """
    Extrai o ID do vídeo de uma URL do YouTube.
    
    Args:
        url: URL do YouTube
        
    Returns:
        ID do vídeo ou None se não encontrado
    """
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    
    match = re.match(youtube_regex, url)
    if match:
        return match.group(6)
    return None


def baixar_video_youtube(
    url: str, 
    output_path: str, 
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Baixa um vídeo do YouTube usando yt-dlp.
    
    Args:
        url: URL do vídeo do YouTube
        output_path: Caminho completo onde o vídeo será salvo
        log_callback: Função opcional para logging de progresso
        
    Returns:
        True se o download foi bem-sucedido, False caso contrário
    """
    def log(msg):
        print(msg)
        if log_callback:
            try:
                log_callback(msg)
            except:
                pass
    
    # Validar URL
    if not validar_url_youtube(url):
        log(f"❌ URL inválida do YouTube: {url}")
        return False
    
    video_id = extrair_video_id(url)
    log(f"📹 Preparando download do vídeo: {video_id}")
    
    # Criar diretório de saída se não existir
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar hook de progresso
    def progress_hook(d):
        if d['status'] == 'downloading':
            try:
                percent_str = d.get('_percent_str', '0%').strip()
                speed_str = d.get('_speed_str', 'N/A').strip()
                eta_str = d.get('_eta_str', 'N/A').strip()
                log(f"⬇️  Baixando: {percent_str} | Velocidade: {speed_str} | ETA: {eta_str}")
            except:
                pass
        elif d['status'] == 'finished':
            log(f"✅ Download concluído! Processando arquivo...")
    
    # Opções do yt-dlp
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress_hook],
        'merge_output_format': 'mp4',
        'postprocessor_args': [
            '-c:v', 'copy',
            '-c:a', 'aac',
        ],
    }
    
    try:
        log(f"🚀 Iniciando download do YouTube...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Obter informações do vídeo
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Vídeo sem título')
            duration = info.get('duration', 0)
            
            minutes = duration // 60
            seconds = duration % 60
            log(f"📝 Título: {title}")
            log(f"⏱️  Duração: {minutes}:{seconds:02d}")
            
            # Fazer download
            ydl.download([url])
        
        # Verificar se o arquivo foi criado
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            log(f"✅ Vídeo baixado com sucesso! ({file_size_mb:.2f} MB)")
            return True
        else:
            log(f"❌ Erro: Arquivo não foi criado em {output_path}")
            return False
            
    except yt_dlp.utils.DownloadError as e:
        log(f"❌ Erro no download: {str(e)}")
        if "Private video" in str(e):
            log("   → Este vídeo é privado e não pode ser baixado")
        elif "Video unavailable" in str(e):
            log("   → Este vídeo não está disponível")
        elif "This video is not available" in str(e):
            log("   → Vídeo não disponível (pode ter restrição geográfica)")
        return False
        
    except Exception as e:
        log(f"❌ Erro inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
