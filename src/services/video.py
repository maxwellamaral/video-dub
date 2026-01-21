
import os
import shutil
import time
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from moviepy.video.fx.MultiplySpeed import MultiplySpeed
from proglog import ProgressBarLogger
from src.config import OUTPUT_DIR

class MyLogger(ProgressBarLogger):
    def __init__(self, custom_callback=None):
        super().__init__()
        self.custom_callback = custom_callback
        self.last_percentage = -1
        self.current_bar = None

    def bars_callback(self, bar, attr, value, old_value=None):
        # Reportar progresso em tempo real
        if self.custom_callback and bar in self.bars:
            total = self.bars[bar]['total']
            if total > 0:
                percentage = int((value / total) * 100)
                # Atualizar apenas quando mudar de porcentagem ou a cada 5%
                if percentage != self.last_percentage and (percentage % 5 == 0 or percentage == 100):
                    bar_name = "Áudio" if bar == 't' else "Vídeo"
                    self.custom_callback(f"   ▸ {bar_name}: {percentage}%")
                    self.last_percentage = percentage

    def callback(self, **changes):
        # Capture text messages from MoviePy
        if self.custom_callback and 'message' in changes:
            msg = changes['message']
            # Filtrar mensagens redundantes
            if 'MoviePy' in msg and 'Done' not in msg:
                self.custom_callback(f"   [FFmpeg] {msg}")

class VideoEditor:
    """
    Gerenciador de edição e manipulação de vídeo.

    Responsável por cortar, redimensionar o tempo (speedup/slowdown) e
    sincronizar o áudio dublado com o vídeo original.
    """
    def __init__(self, caminho_video):
        """
        Carrega o vídeo original usando MoviePy.

        Args:
            caminho_video (str): Path do arquivo de vídeo.
        """
        self.caminho_video = caminho_video
        try:
            self.video_original = VideoFileClip(caminho_video)
            self.fps = self.video_original.fps
            self.duration = self.video_original.duration
        except Exception as e:
            print(f"✗ Erro ao abrir vídeo: {e}")
            raise e
            
    def close(self):
        if hasattr(self, 'video_original') and self.video_original:
            self.video_original.close()
            
    def processar_segmentos(self, segmentos, audios_sintetizados, log_callback=None):
        """
        Gera uma lista de videoclips sincronizados com o novo áudio.

        Ajusta a velocidade do vídeo (time stretching) para casar com a duração
        do áudio dublado, dentro de limites aceitáveis (0.1x a 10x).

        Args:
            segmentos (list): Lista de legendas traduzidas (metadata).
            audios_sintetizados (list): Lista de áudios (numpy arrays).
            log_callback (callable, optional): Função para logar mensagens.

        Returns:
            tuple: (lista_clips_video, lista_arquivos_temp, novas_legendas)
        """
        msg = f"   🎬 Sincronizando {len(segmentos)} segmentos..."
        if log_callback: log_callback(msg)
        else: print(msg)
        
        clips_finais = []
        arquivos_temp = []
        novas_legendas = []
        tempo_acumulado = 0.0
        
        for i, seg in enumerate(segmentos):
            if i >= len(audios_sintetizados): break
            audio_data, sr = audios_sintetizados[i]
            
            start_t = seg["start"]
            end_t = min(seg["end"], self.duration)
            original_dur = end_t - start_t
            
            if start_t >= self.duration: break
            if original_dur <= 0.1: continue
            
            # Recorte inicial
            clip = self.video_original.subclipped(start_t, end_t)
            final_dur = original_dur
            
            # Se tem áudio sintentizado
            if audio_data is not None and len(audio_data) > 0:
                # Padding de segurança (200ms)
                padding = int(sr * 0.2)
                audio_padded = np.pad(audio_data, (0, padding), mode='constant')
                
                temp_wav = os.path.join(OUTPUT_DIR, f"temp_seg_{i}.wav")
                sf.write(temp_wav, audio_padded, int(sr))
                arquivos_temp.append(temp_wav)
                
                audio_clip = AudioFileClip(temp_wav)
                
                # Calcular speedup/slowdown
                # Usar duração real do áudio (sem padding excessivo) para calcular ratio
                # O clip de áudio final terá duration = final_dur
                audio_dur = len(audio_data) / sr
                
                ratio = original_dur / audio_dur
                ratio = max(0.1, min(ratio, 10.0)) # Clamp
                
                if abs(ratio - 1.0) > 0.05:
                    # Ajustar velocidade do vídeo
                    clip = clip.with_effects([MultiplySpeed(ratio)])
                    final_dur = original_dur / ratio # Novo tempo = Dist / Vel
                else:
                    final_dur = audio_dur
                    
                # Fixar áudio
                audio_clip = audio_clip.with_duration(final_dur)
                clip = clip.with_audio(audio_clip)
                clip = clip.with_duration(final_dur)
                
            else:
                clip = clip.without_audio()
                
            # Padronizar
            clip = clip.with_fps(self.fps)
            clips_finais.append(clip)
            
            novas_legendas.append({
                "start": tempo_acumulado,
                "end": tempo_acumulado + final_dur,
                "text": seg["text"]
            })
            tempo_acumulado += final_dur
            
        return clips_finais, arquivos_temp, novas_legendas

    def renderizar_video(self, clips, caminho_saida, modo="rapido", log_callback=None):
        """
        Compila a lista de clips finais em um único arquivo de vídeo.

        Args:
            clips (list): Lista de objetos VideoFileClip processados.
            caminho_saida (str): Path final do arquivo .mp4.
            modo (str): 'rapido' (NVENC) ou 'qualidade' (libx264).
            log_callback (callable, optional): Função para logar mensagens.

        Returns:
            bool: True se sucesso.
        """
        if not clips: return False
        
        msg = "   Concatenando clips..."
        if log_callback: log_callback(msg)
        else: print(msg)

        # Validar FPS
        clips = [c.with_fps(24) if not c.fps else c for c in clips]
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Parâmetros de Encoding
        params_gpu = {
            "codec": "h264_nvenc",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "fps": 24,
            "preset": "p1",
            "ffmpeg_params": ["-rc", "vbr", "-cq", "23", "-b:v", "0"]
        }
        
        params_cpu = {
            "codec": "libx264",
            "audio_codec": "aac",
            "audio_bitrate": "192k",
            "temp_audiofile": "temp-audio.m4a",
            "remove_temp": True,
            "fps": 24,
            "preset": "medium",
            "threads": 4,
            "ffmpeg_params": ["-crf", "18"]
        }
        
        success = False
        start_t = time.time()
        
        # Prepare Logger
        logger = "bar" # Default
        if log_callback:
            logger = MyLogger(log_callback)

        if modo == "rapido":
            msg_gpu = "   🚀 Renderizando (GPU NVENC)..."
            if log_callback: log_callback(msg_gpu)
            else: print(msg_gpu)

            try:
                final_video.write_videofile(caminho_saida, **params_gpu, logger=logger)
                success = True
            except Exception as e:
                msg_fail = f"   ⚠️ Falha GPU: {e}. Tentando CPU..."
                if log_callback: log_callback(msg_fail)
                else: print(msg_fail)
                # Fallback para CPU logic abaixo
        
        if not success: # Se não era rápido ou se falhou
            msg_cpu = "   🎬 Renderizando (CPU libx264)..."
            if log_callback: log_callback(msg_cpu)
            else: print(msg_cpu)
            final_video.write_videofile(caminho_saida, **params_cpu, logger=logger)
            
        render_time = f"   ⏱️ Tempo render: {time.time() - start_t:.1f}s"
        if log_callback: log_callback(render_time)
        else: print(render_time)

        final_video.close()
        for c in clips: c.close()
        return True
