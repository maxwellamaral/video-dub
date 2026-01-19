
import os
import shutil
import time
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from src.config import OUTPUT_DIR

class VideoEditor:
    def __init__(self, caminho_video):
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
            
    def processar_segmentos(self, segmentos, audios_sintetizados):
        """
        Gera clips sincronizados.
        audios_sintetizados: lista de (audio_numpy, sample_rate)
        """
        print(f"   🎬 Sincronizando {len(segmentos)} segmentos...")
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
                    clip = clip.time_transform(lambda t: t * ratio)
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

    def renderizar_video(self, clips, caminho_saida, modo="rapido"):
        """Compila e salva o vídeo final."""
        if not clips: return False
        
        print("   Concatenando clips...")
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
        
        if modo == "rapido":
            print("   🚀 Renderizando (GPU NVENC)...")
            try:
                final_video.write_videofile(caminho_saida, **params_gpu, logger="bar")
                success = True
            except Exception as e:
                print(f"   ⚠️ Falha GPU: {e}. Tentando CPU...")
                # Fallback para CPU logic abaixo
        
        if not success: # Se não era rápido ou se falhou
            print("   🎬 Renderizando (CPU libx264)...")
            final_video.write_videofile(caminho_saida, **params_cpu, logger="bar")
            
        print(f"   ⏱️ Tempo render: {time.time() - start_t:.1f}s")
        final_video.close()
        for c in clips: c.close()
        return True
