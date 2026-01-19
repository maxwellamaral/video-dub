# ============================================================================
# PIPELINE DE DUBLAGEM DE VÍDEOS COM HUGGING FACE
# Transcrição → Tradução → Síntese de Voz → Remontagem
# Otimizado para GPU NVIDIA
# ============================================================================

import os
import sys
import subprocess
from pathlib import Path

# Detecção precoce do ffmpeg para evitar problemas de PATH no Windows
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Adicionar o diretório do ffmpeg ao PATH para que o Whisper o encontre
    ffmpeg_dir = os.path.dirname(FFMPEG_EXE)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    
    # Garantir que exista um arquivo chamado ffmpeg.exe no diretório (algumas bibliotecas exigem esse nome exato)
    ffmpeg_base = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_base):
        import shutil
        try:
            shutil.copy(FFMPEG_EXE, ffmpeg_base)
            print(f"✓ Criado link de compatibilidade: {ffmpeg_base}")
        except Exception as e:
            print(f"⚠️ Aviso ao criar link de ffmpeg: {e}")

except Exception as e:
    print(f"⚠️ Aviso: imageio_ffmpeg falhou: {e}")
    FFMPEG_EXE = "ffmpeg"

print(f"✓ FFMPEG Path: {FFMPEG_EXE}")

import torch
import torchaudio
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import VitsModel, AutoTokenizer as TTSTokenizer
import numpy as np
import soundfile as sf
import librosa
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# Novos motores TTS
import kokoro_onnx
from TTS.api import TTS

# ============================================================================
# CONFIGURAÇÕES INICIAIS
# ============================================================================

# Defina o dispositivo: "cuda:0" para GPU ou "cpu"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"✓ Usando dispositivo: {DEVICE}")

# Idiomas suportados pelo NLLB (formato: {idioma}_{script})
# Exemplos: eng_Latn (inglês), por_Latn (português), spa_Latn (espanhol)
# Lista completa: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
IDIOMA_ORIGEM = "eng_Latn"      # Inglês
IDIOMA_DESTINO = "por_Latn"    # Português

# Caminhos
INPUT_DIR = "input"
os.makedirs(INPUT_DIR, exist_ok=True)
VIDEO_ENTRADA = os.path.join(INPUT_DIR, "video_entrada.mp4")      # Seu vídeo

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIO_EXTRAIDO = os.path.join(OUTPUT_DIR, "audio_extraido.wav")
AUDIO_REFERENCIA = os.path.join(OUTPUT_DIR, "referencia_voz.wav")
AUDIO_TRADUZIDO = os.path.join(OUTPUT_DIR, "audio_traduzido.wav")
VIDEO_SAIDA_BASE = os.path.join(OUTPUT_DIR, "video_dublado") # Sufixo será adicionado
LEGENDA_ORIGINAL = os.path.join(OUTPUT_DIR, "legenda_original.srt")
LEGENDA_TRADUZIDA = os.path.join(OUTPUT_DIR, "legenda_traduzida.srt")

# Motores TTS Disponíveis
MOTORES_TTS = ["mms", "coqui"]

def obter_ffmpeg_exe():
    """Retorna o caminho do executável ffmpeg."""
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except:
        return "ffmpeg" # Fallback para o PATH

# ============================================================================
# ETAPA 1: EXTRAÇÃO DE ÁUDIO DO VÍDEO (usando ffmpeg)
# ============================================================================

def extrair_referencia_voz(caminho_video, caminho_saida, duracao=10):
    """
    Extrai os primeiros segundos do vídeo original para usar como referência de clonagem.
    """
    print(f"🎙️ Extraindo referência de voz ({duracao}s) para clonagem...")
    video = None
    try:
        from moviepy import VideoFileClip
        video = VideoFileClip(caminho_video)
        # Extrair áudio dos primeiros 10 segundos
        trecho = video.subclipped(0, min(duracao, video.duration))
        trecho.audio.write_audiofile(caminho_saida, fps=22050, nbytes=2, codec='pcm_s16le')
        print(f"✓ Referência salva em: {caminho_saida}")
        return True
    except Exception as e:
        print(f"⚠️ Erro ao extrair referência de voz: {e}")
        return False
    finally:
        if video:
            try:
                video.close()
            except:
                pass

def extrair_audio(caminho_video, caminho_audio_saida):
    """
    Extrai áudio do vídeo usando ffmpeg.
    Certifique-se de ter ffmpeg instalado: apt-get install ffmpeg (Linux/WSL)
    """
    print(f"\n📹 Extraindo áudio de: {caminho_video}")
    try:
        cmd = [
            FFMPEG_EXE, "-i", caminho_video,
            "-q:a", "9", "-n",  # -n não sobrescreve
            caminho_audio_saida
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Áudio extraído: {caminho_audio_saida}")
        return True
    except Exception as e:
        print(f"✗ Erro ao extrair áudio: {e}")
        return False

# ============================================================================
# ETAPA 2: TRANSCRIÇÃO COM WHISPER
# ============================================================================

def transcrever_audio(caminho_audio):
    """
    Transcreve áudio para texto usando Whisper com ALTA GRANULARIDADE.
    Retorna uma lista de segmentos otimizados para legendas.
    """
    print(f"\n🎙️  Transcrevendo áudio com Whisper (word-level timestamps)...")
    
    # Usar modelo slightly maior para melhor timestamp se possível, mas base funciona
    modelo_whisper = "openai/whisper-base"
    
    pipe_speech = pipeline(
        task="automatic-speech-recognition",
        model=modelo_whisper,
        device=0 if DEVICE == "cuda:0" else -1,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32,
        chunk_length_s=30,
    )
    
    # Solicitar timestamps por palavra para maior precisão
    # Nota: nem todos os modelos/versões suportam word level perfeitamente,
    # mas o pipeline do transformers mais recente costuma suportar.
    try:
        resultado = pipe_speech(caminho_audio, return_timestamps="word")
    except Exception as e:
        print(f"⚠️  Aviso: 'word' timestamps falhou ({e}), tentando 'True' padrão...")
        resultado = pipe_speech(caminho_audio, return_timestamps=True)

    # Extrair palavras e tempos
    raw_chunks = resultado.get("chunks", [])
    if not raw_chunks:
        # Tentar pegar do text se não houver chunks
        raw_chunks = [{"text": resultado.get("text", ""), "timestamp": (0.0, 0.0)}]

    # --- Reagrupar palavras em segmentos de legenda ---
    segmentos_finais = []
    
    # Configurações de agrupamento
    MAX_CHARS_POR_SEGMENTO = 80    # Máximo de caracteres por legenda
    MAX_DURACAO_SEGMENTO = 7.0     # Máximo de segundos por legenda
    MIN_PAUSA_QUEBRA = 0.5         # Pausa que força nova legenda
    
    buffer_palavras = []
    start_time = 0.0
    last_end_time = 0.0
    buffer_text_len = 0
    
    # Se raw_chunks vier vazio ou estranho, garantir robustez
    for i, chunk in enumerate(raw_chunks):
        # O formato do timestamp pode ser (start, end) ou dicionário
        times = chunk.get("timestamp")
        text = chunk.get("text", "").strip()
        
        if not text:
            continue
            
        if isinstance(times, (list, tuple)):
            c_start, c_end = times
        else:
            c_start, c_end = last_end_time, last_end_time + 1.0

        if c_start is None: c_start = last_end_time
        if c_end is None: c_end = c_start + 0.3 # estimativa
        
        # Inicializar primeiro segmento
        if not buffer_palavras and not segmentos_finais:
            start_time = c_start
            
        # Calcular pausas
        pausa_anterior = c_start - last_end_time
        tempo_decorrido = c_end - start_time
        
        should_break = False
        
        # Lógica de quebra
        # 1. Se acabou de começar um novo buffer
        if not buffer_palavras:
             start_time = c_start
             
        # Critérios:
        if buffer_palavras:
             # Pausa longa
             if pausa_anterior > MIN_PAUSA_QUEBRA:
                 should_break = True
             # Duração excessiva
             elif tempo_decorrido > MAX_DURACAO_SEGMENTO:
                 should_break = True
             # Texto muito longo
             elif buffer_text_len + len(text) > MAX_CHARS_POR_SEGMENTO:
                 should_break = True
             # Pontuação forte
             elif buffer_palavras[-1][-1] in ".?!":
                 should_break = True
        
        if should_break:
            texto_seg = " ".join(buffer_palavras)
            segmentos_finais.append({
                "start": start_time,
                "end": last_end_time,
                "text": texto_seg
            })
            buffer_palavras = []
            start_time = c_start
            buffer_text_len = 0
            
        buffer_palavras.append(text)
        buffer_text_len += len(text) + 1
        last_end_time = c_end
        
    # Adicionar o buffer restante
    if buffer_palavras:
        texto_seg = " ".join(buffer_palavras)
        segmentos_finais.append({
            "start": start_time,
            "end": last_end_time,
            "text": texto_seg
        })
        
    # Se ainda assim não gerou nada, usar o texto bruto
    if not segmentos_finais:
         segmentos_finais.append({
            "start": 0.0,
            "end": last_end_time if last_end_time > 0 else 1.0,
            "text": resultado.get("text", "")
        })

    print(f"✓ Transcrição granular: {len(segmentos_finais)} segmentos gerados.")
    return segmentos_finais

# ============================================================================
# ETAPA 3: TRADUÇÃO COM NLLB (No Language Left Behind)
# ============================================================================

def traduzir_segmentos(segmentos, idioma_origem=IDIOMA_ORIGEM, idioma_destino=IDIOMA_DESTINO):
    """
    Traduz segmentos mantendo os timestamps.
    
    Args:
        segmentos: Lista de dicts com {'start': float, 'end': float, 'text': str}
        idioma_origem: Código do idioma de origem (NLLB format)
        idioma_destino: Código do idioma de destino (NLLB format)
        
    Retorna:
        list: Lista de dicts traduzidos com mesmos timestamps
    """
    print(f"\n🌐 Traduzindo de {idioma_origem} para {idioma_destino}...")
    
    pipe_translation = pipeline(
        task="translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=idioma_origem,
        tgt_lang=idioma_destino,
        device=0 if DEVICE == "cuda:0" else -1,
        torch_dtype=torch.float16 if "cuda" in DEVICE else torch.float32
    )
    
    segmentos_traduzidos = []
    total = len(segmentos)
    
    print(f"   Traduzindo {total} segmentos...")
    
    for i, seg in enumerate(segmentos):
        texto = seg["text"].strip()
        if not texto:
            continue
            
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Segmento {i+1}/{total}")
        
        try:
            resultado = pipe_translation(texto, max_length=512)
            texto_traduzido = resultado[0]["translation_text"]
            
            segmentos_traduzidos.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": texto_traduzido
            })
        except Exception as e:
            print(f"   ⚠️  Erro no segmento {i+1}: {e}")
            # Manter o texto original em caso de erro
            segmentos_traduzidos.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": texto
            })
    
    # Calcular estatísticas
    texto_final = " ".join([s["text"] for s in segmentos_traduzidos])
    print(f"✓ Texto traduzido ({len(texto_final)} caracteres, {len(segmentos_traduzidos)} segmentos)")
    print(f"   {texto_final[:100]}...")
    
    return segmentos_traduzidos


def segmentos_para_srt(segmentos):
    """
    Converte lista de segmentos para formato SRT.
    
    Args:
        segmentos: Lista de dicts com {'start': float, 'end': float, 'text': str}
        
    Retorna:
        str: Conteúdo do arquivo SRT
    """
    def formatar_tempo_srt(segundos):
        """Converte segundos para formato SRT (HH:MM:SS,mmm)"""
        if segundos is None or segundos < 0:
            segundos = 0
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        secs = int(segundos % 60)
        millis = int((segundos % 1) * 1000)
        return f"{horas:02d}:{minutos:02d}:{secs:02d},{millis:03d}"
    
    linhas = []
    for i, seg in enumerate(segmentos, 1):
        inicio = formatar_tempo_srt(seg["start"])
        fim = formatar_tempo_srt(seg["end"])
        texto = seg["text"].strip()
        
        if texto:  # Só adicionar se houver texto
            linhas.append(f"{i}")
            linhas.append(f"{inicio} --> {fim}")
            linhas.append(texto)
            linhas.append("")  # Linha em branco entre legendas
    
    return "\n".join(linhas)


def segmentos_para_texto(segmentos):
    """
    Extrai apenas o texto dos segmentos (para síntese de voz).
    
    Args:
        segmentos: Lista de dicts com {'start': float, 'end': float, 'text': str}
        
    Retorna:
        str: Texto concatenado
    """
    return " ".join([s["text"] for s in segmentos if s["text"].strip()])

# ============================================================================
# ETAPA 4: SÍNTESE DE VOZ (TEXT-TO-SPEECH) COM MMS-TTS
# ============================================================================

def sintetizar_voz(texto, idioma="por"):
    """
    Sintetiza texto em voz usando MMS-TTS do Facebook.
    
    Idiomas suportados (códigos ISO 639-3):
    - por: português
    - eng: inglês  
    - spa: espanhol
    - fra: francês
    - deu: alemão
    
    Modelos: facebook/mms-tts-{idioma}
    """
    print(f"\n🔊 Sintetizando fala em {idioma}...")
    
    modelo_nome = f"facebook/mms-tts-{idioma}"
    
    try:
        model = VitsModel.from_pretrained(modelo_nome)
        model = model.to(DEVICE)
        tokenizer = TTSTokenizer.from_pretrained(modelo_nome)
        
        # Limpar o texto - remover caracteres especiais problemáticos
        import re
        texto_limpo = texto
        # Remover caracteres de controle e caracteres não-imprimíveis
        texto_limpo = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', texto_limpo)
        # Substituir múltiplos espaços por um único
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo)
        # Remover caracteres especiais que podem causar problemas
        texto_limpo = re.sub(r'[^\w\s.,!?;:\-\'\"áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ]', '', texto_limpo)
        texto_limpo = texto_limpo.strip()
        
        if not texto_limpo:
            raise ValueError("Texto vazio após limpeza")
        
        print(f"   Texto original: {len(texto)} chars, limpo: {len(texto_limpo)} chars")
        
        # Dividir por pontuação de fim de sentença
        sentences = re.split(r'(?<=[.!?])\s+', texto_limpo)
        
        # Agrupar sentenças em chunks de no máximo 200 caracteres
        max_chars = 200
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Se a sentença sozinha for muito grande, dividir por palavras
            if len(sentence) > max_chars:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chars:
                        temp_chunk = (temp_chunk + " " + word).strip()
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    chunks.append(temp_chunk)
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if not chunks:
            chunks = [texto_limpo[:max_chars]]  # Fallback
            
        print(f"   Processando {len(chunks)} chunks de texto...")
        
        all_audio = []
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk or len(chunk) < 3:
                continue
            print(f"   Chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
            try:
                inputs = tokenizer(text=chunk, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    output = model(**inputs).waveform
                
                all_audio.append(output.cpu().numpy()[0])
            except Exception as chunk_error:
                print(f"   ⚠️  Erro no chunk {i+1}, pulando: {chunk_error}")
                continue
        
        if not all_audio:
            raise ValueError("Nenhum áudio gerado - todos os chunks falharam")
            
        # Concatenar todos os chunks
        audio_numpy = np.concatenate(all_audio)
        
        print(f"✓ Áudio sintetizado ({len(audio_numpy)} amostras)")
        
        return audio_numpy, model.config.sampling_rate
    
    except Exception as e:
        print(f"✗ Erro na síntese de voz: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def salvar_audio(audio_numpy, sample_rate, caminho_saida):
    """
    Salva array numpy como arquivo WAV.
    """
    try:
        # Se for mono (1D), reshapear
        if audio_numpy.ndim == 1:
            audio_numpy = np.expand_dims(audio_numpy, axis=0)
        
        # Converter para tensor PyTorch e salvar
        waveform = torch.from_numpy(audio_numpy).float()
        
        torchaudio.save(
            caminho_saida,
            waveform,
            sample_rate=int(sample_rate)
        )
        print(f"✓ Áudio salvo em: {caminho_saida}")
        return True
    except Exception as e:
        print(f"✗ Erro ao salvar áudio: {e}")
        return False

def sintetizar_segmento_audio(texto, motor, config):
    """
    Sintetiza áudio usando o motor escolhido (mms, kokoro, coqui).
    Retorna (audio_data_numpy, sample_rate).
    """
    if motor == "mms":
        model = config["model"]
        tokenizer = config["tokenizer"]
        device = config["device"]
        
        # Limpeza básica para MMS
        texto_limpo = "".join([c for c in texto if c.isalnum() or c in " ,.?!"])
        inputs = tokenizer(texto_limpo, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(**inputs).waveform
        
        audio_np = output.cpu().numpy().squeeze()
        return audio_np, model.config.sampling_rate

    elif motor == "coqui":
        tts = config["tts"]
        ref_wav = config["ref_wav"]
        lang_code = config.get("lang_coqui", "pt")
        
        # XTTS retorna lista de floats
        wav = tts.tts(text=texto, speaker_wav=ref_wav, language=lang_code)
        return np.array(wav), 24000

    return None, None

def dublar_com_ajuste_video(caminho_video, segmentos, idioma_voz, saida_video, motor_tts="mms"):
    """
    Versão aprimorada que suporta múltiplos motores TTS.
    """
    print(f"\n️ Iniciando síntese e sincronização com motor: {motor_tts.upper()}")
    
    config = {"device": DEVICE, "idioma_voz": idioma_voz}
    sample_rate = None # Será definido após carregar o modelo
    
    # 1. Inicializar Motor escolhido
    try:
        if motor_tts == "mms":
            modelo_nome = f"facebook/mms-tts-{idioma_voz}"
            print(f"   Carregando MMS-TTS: {modelo_nome}")
            from transformers import VitsModel, AutoTokenizer
            config["tokenizer"] = AutoTokenizer.from_pretrained(modelo_nome)
            config["model"] = VitsModel.from_pretrained(modelo_nome).to(DEVICE)
            sample_rate = config["model"].config.sampling_rate
            
        elif motor_tts == "coqui":
            print("   Carregando Coqui XTTS v2 (pode demorar na primeira vez)...")
            # Aceitar licença automaticamente para automação
            os.environ["COQUI_TOS_AGREED"] = "1"
            try:
                from TTS.api import TTS
                import torch
                
                # Workaround para torch.load weights_only no PyTorch 2.6+
                # Temporariamente desabilitar weights_only para carregar modelos Coqui confiáveis
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load
                
                print("      Inicializando modelo XTTS v2...")
                config["tts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                
                # Restaurar torch.load original
                torch.load = original_load
                
                print("      Movendo modelo para GPU...")
                config["tts"] = config["tts"].to(DEVICE)
                config["ref_wav"] = AUDIO_REFERENCIA
                config["lang_coqui"] = "pt" # XTTS usa 'pt' para português
                sample_rate = 24000 # XTTS v2 geralmente usa 24000 Hz
                print("      ✓ Coqui XTTS v2 carregado com sucesso!")
            except Exception as coqui_err:
                print(f"✗ Erro ao carregar Coqui TTS: {coqui_err}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"✗ Motor TTS '{motor_tts}' não suportado.")
            return False
        
        if sample_rate is None:
            print("✗ Não foi possível determinar o sample_rate do motor TTS.")
            return False

    except Exception as e:
        print(f"✗ Erro ao carregar modelo TTS: {e}")
        import traceback
        traceback.print_exc()
        return False

    clips_finais = []
    
    try:
        video_original = VideoFileClip(caminho_video)
        fps_original = video_original.fps  # Capturar fps do vídeo original
        print(f"   FPS do vídeo original: {fps_original}")
    except Exception as e:
        print(f"✗ Erro ao abrir vídeo original: {e}")
        return False
        
    print(f"   Processando {len(segmentos)} segmentos...")
    
    # Processar cada segmento
    clips_finais = []
    tempo_acumulado = 0.0
    novos_segmentos_legenda = []

    for i, seg in enumerate(segmentos):
        texto = seg["text"]
        start_t = seg["start"]
        end_t = seg["end"]
        
        # Síntese
        audio_data, sr = sintetizar_segmento_audio(texto, motor_tts, config)
        
        if audio_data is None:
            duracao_audio = 0.5 
            audio_clip = None
        else:
            temp_wav = os.path.join(OUTPUT_DIR, f"temp_seg_{i}.wav")
            import soundfile as sf
            sf.write(temp_wav, audio_data, int(sr)) # Use sr from sintetizar_segmento_audio
            audio_clip = AudioFileClip(temp_wav)
            duracao_audio = audio_clip.duration
            
        if start_t >= video_original.duration:
            break
        end_t = min(end_t, video_original.duration)
        
        duracao_video_orig = end_t - start_t
        if duracao_video_orig <= 0.1: continue
        
        video_clip = video_original.subclipped(start_t, end_t)
        
        if audio_clip:
            ratio = duracao_video_orig / duracao_audio
            ratio = max(0.1, min(ratio, 10.0)) 
            
            if abs(ratio - 1.0) > 0.05:
                original_fps = video_clip.fps
                # Usar time_transform para ajustar velocidade no MoviePy 2.x
                video_clip = video_clip.time_transform(lambda t: t * ratio)
                # Preservar fps após transformação
                if original_fps and not video_clip.fps:
                    video_clip.fps = original_fps
                # Calcular a nova duração após transformação (time_transform não preserva duration)
                duracao_final_clip = duracao_video_orig / ratio
            else:
                duracao_final_clip = duracao_video_orig
            
            video_clip = video_clip.with_audio(audio_clip)
        else:
            video_clip = video_clip.without_audio()
            duracao_final_clip = duracao_video_orig
            
        # Fallback: se ainda for None, usar duração original do segmento
        if duracao_final_clip is None:
            duracao_final_clip = duracao_video_orig
        
        # Garantir que o clip tenha o fps correto do vídeo original
        if not video_clip.fps or video_clip.fps != fps_original:
            video_clip = video_clip.with_fps(fps_original)
        
        # Garantir que o clip tenha duração definida explicitamente (MoviePy 2.x)
        video_clip = video_clip.with_duration(duracao_final_clip)
        
        clips_finais.append(video_clip)
        
        # Guardar novo timestamp para legenda
        novos_segmentos_legenda.append({
            "start": tempo_acumulado,
            "end": tempo_acumulado + duracao_final_clip,
            "text": texto
        })
        tempo_acumulado += duracao_final_clip
        
        if (i+1) % 10 == 0:
            print(f"   Seg {i+1}/{len(segmentos)} processado.")

    print("   Concatenando clips e salvando (isso pode demorar)...")
    try:
        # Garantir que todos os clips tenham fps definido usando set_fps
        clips_com_fps = []
        for clip in clips_finais:
            if not clip.fps or clip.fps is None:
                clip = clip.set_fps(24)  # Usar set_fps method do MoviePy
            clips_com_fps.append(clip)
        
        final_video = concatenate_videoclips(clips_com_fps, method="compose")
        final_video.write_videofile(saida_video, codec="libx264", audio_codec="aac", fps=24, logger="bar")
        
        # Salvar nova legenda sincronizada
        try:
            srt_final = segmentos_para_srt(novos_segmentos_legenda)
            nome_legenda_final = os.path.join(OUTPUT_DIR, "legenda_final_sincronizada.srt")
            with open(nome_legenda_final, "w", encoding="utf-8") as f:
                f.write(srt_final)
            print(f"✓ Legenda final sincronizada salva em: {nome_legenda_final}")
        except Exception as e_leg:
            print(f"⚠️ Erro ao salvar legenda final: {e_leg}")
        
        import glob
        for f in glob.glob(os.path.join(OUTPUT_DIR, "temp_seg_*.wav")):
            try: os.remove(f)
            except: pass
        
        # Fechar recursos para evitar WinError 6 no Windows
        try:
            final_video.close()
        except:
            pass
        for clip in clips_finais:
            try:
                clip.close()
            except:
                pass
            
        return True
    except Exception as e:
        print(f"✗ Erro na montagem final: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Garantir fechamento do vídeo original para evitar WinError 6
        try:
            video_original.close()
        except:
            pass

# ============================================================================
# ETAPA 5: REMONTAGEM DE VÍDEO COM NOVO ÁUDIO (ffmpeg)
# ============================================================================

def remontar_video(caminho_video_orig, caminho_audio_novo, caminho_saida):
    """
    Substitui a faixa de áudio original pelo áudio dublado.
    Usa ffmpeg para muxar vídeo + novo áudio.
    """
    print(f"\n🎬 Remontando vídeo com áudio dublado...")
    
    try:
        cmd = [
            FFMPEG_EXE, "-i", caminho_video_orig,
            "-i", caminho_audio_novo,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
            "-n",                     # Não sobrescrever
            caminho_saida
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Vídeo dublado salvo em: {caminho_saida}")
        return True
    except Exception as e:
        print(f"✗ Erro ao remontar vídeo: {e}")
        return False

# ============================================================================
# PIPELINE COMPLETA (ORQUESTRAÇÃO)
# ============================================================================

def executar_pipeline_completa(
    caminho_video,
    idioma_origem="eng_Latn",
    idioma_destino="por_Latn",
    idioma_voz="por",
    motor_tts="mms"
):
    """
    Executa a pipeline completa de dublagem.
    """
    print("=" * 70)
    print(f"INICIANDO PIPELINE DE DUBLAGEM ({motor_tts.upper()})")
    print("=" * 70)
    
    # Definir nome de saída com base no motor
    saida_video = f"{VIDEO_SAIDA_BASE}_{motor_tts}.mp4"
    
    # 0. Extrair referência de voz se for Coqui (fazemos antes para falhar cedo se necessário)
    if motor_tts == "coqui":
        if not extrair_referencia_voz(caminho_video, AUDIO_REFERENCIA):
            print("✗ Falha ao obter voz de referência. Abortando.")
            return False

    # 1. Extrair áudio original
    if not extrair_audio(caminho_video, AUDIO_EXTRAIDO):
        print("✗ Falha na extração de áudio. Abortando.")
        return False
    
    # 2. Transcrever (retorna segmentos)
    segmentos_originais = transcrever_audio(AUDIO_EXTRAIDO)
    if not segmentos_originais:
        print("✗ Falha na transcrição. Abortando.")
        return False
    
    # 2.1. Salvar legenda original (SRT) para conferência
    try:
        conteudo_srt = segmentos_para_srt(segmentos_originais)
        with open(LEGENDA_ORIGINAL, "w", encoding="utf-8") as f:
            f.write(conteudo_srt)
        print(f"✓ Legenda original salva em: {LEGENDA_ORIGINAL}")
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível salvar legenda original: {e}")
    
    # 3. Traduzir segmentos
    segmentos_traduzidos = traduzir_segmentos(segmentos_originais, idioma_origem, idioma_destino)
    if not segmentos_traduzidos:
        print("✗ Falha na tradução. Abortando.")
        return False
    
    # 3.1. Salvar legenda traduzida (SRT) preliminar
    try:
        conteudo_srt_trad = segmentos_para_srt(segmentos_traduzidos)
        with open(LEGENDA_TRADUZIDA, "w", encoding="utf-8") as f:
            f.write(conteudo_srt_trad)
        print(f"✓ Legenda traduzida salva em: {LEGENDA_TRADUZIDA}")
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível salvar legenda traduzida: {e}")

    # 4. Sintetizar e Montar com ajuste dinâmico de vídeo
    if not dublar_com_ajuste_video(caminho_video, segmentos_traduzidos, idioma_voz, saida_video, motor_tts):
        print("✗ Falha na síntese/montagem final. Abortando.")
        return False
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETADA COM SUCESSO!")
    print(f"✓ Vídeo dublado salvo em: {saida_video}")
    print(f"✓ Legenda original salva em: {LEGENDA_ORIGINAL}")
    print(f"✓ Legenda final sincronizada: {os.path.join(OUTPUT_DIR, 'legenda_final_sincronizada.srt')}")
    print("=" * 70)
    
    return True

def obter_configuracao_usuario():
    """Exibe um menu para o usuário escolher os idiomas e o motor de dublagem."""
    opcoes_idioma = [
        {"nome": "Inglês para Português (Brasil)", "origem": "eng_Latn", "destino": "por_Latn", "voz": "por"},
        {"nome": "Português para Inglês", "origem": "por_Latn", "destino": "eng_Latn", "voz": "eng"},
        {"nome": "Espanhol para Português (Brasil)", "origem": "spa_Latn", "destino": "por_Latn", "voz": "por"},
        {"nome": "Personalizado", "origem": "manual", "destino": "manual", "voz": "manual"},
        {"nome": "Sair", "origem": "exit", "destino": "exit", "voz": "exit"}
    ]

    print("\n" + "="*50)
    print("       1. CONFIGURAÇÃO DE IDIOMAS")
    print("="*50)
    for i, opcao in enumerate(opcoes_idioma, 1):
        print(f"{i}. {opcao['nome']}")
    
    try:
        escolha_idioma = int(input("\nEscolha os idiomas (padrão 1): ") or "1")
        config_id = opcoes_idioma[escolha_idioma-1]
        
        if config_id["origem"] == "exit": return None, None, None, None
        
        origem, destino, voz = config_id["origem"], config_id["destino"], config_id["voz"]
        if origem == "manual":
            origem = input("Código NLLB Origem (ex: eng_Latn): ") or "eng_Latn"
            destino = input("Código NLLB Destino (ex: por_Latn): ") or "por_Latn"
            voz = input("Código MMS-TTS Voz (ex: por): ") or "por"

    except (ValueError, IndexError):
        origem, destino, voz = "eng_Latn", "por_Latn", "por"

    print("\n" + "="*50)
    print("       2. MECANISMO DE VOZ (TTS)")
    print("="*50)
    print("1. MMS-TTS (Padrão, Offline, Leve)")
    print("2. Coqui XTTS v2 (Clonagem de Voz do Vídeo)")
    
    try:
        escolha_tts = int(input("\nEscolha o motor (padrão 1): ") or "1")
        motores = {1: "mms", 2: "coqui"}
        motor = motores.get(escolha_tts, "mms")
    except ValueError:
        motor = "mms"

    return origem, destino, voz, motor

if __name__ == "__main__":
    # Certifique-se de que seu vídeo existe
    if not os.path.exists(VIDEO_ENTRADA):
        print(f"\n[!] Arquivo não encontrado: {VIDEO_ENTRADA}")
        print(f"    Por favor, coloque seu vídeo na pasta '{INPUT_DIR}' e renomeie para 'video_entrada.mp4'")
    else:
        # Obter configurações
        origem, destino, voz, motor = obter_configuracao_usuario()
        
        if origem is None:
            print("\n👋 Saindo...")
            exit()
        
        # Executar pipeline completa
        sucesso = executar_pipeline_completa(
            caminho_video=VIDEO_ENTRADA,
            idioma_origem=origem,
            idioma_destino=destino,
            idioma_voz=voz,
            motor_tts=motor
        )
        
        if sucesso:
            print("\n💡 Dicas:")
            print("  • Para vídeos mais longos, considere usar modelo 'tiny' no Whisper")
            print("  • Se ficar sem memória, reduza para float16 ou use modelos menores")
            print("  • Edite os códigos de idioma conforme necessário no topo do arquivo")

# ============================================================================
# REFERÊNCIAS DE IDIOMAS (NLLB e MMS-TTS)
# ============================================================================

# Códigos NLLB (Flores-200 - amostra):
# eng_Latn = inglês
# por_Latn = português
# spa_Latn = espanhol
# fra_Latn = francês
# deu_Latn = alemão
# ita_Latn = italiano
# jpn_Jpan = japonês
# zho_Hans = chinês simplificado
# rus_Cyrl = russo
# ara_Arab = árabe

# Idiomas MMS-TTS (ISO 639-3):
# por = português
# eng = inglês
# spa = espanhol
# fra = francês
# deu = alemão
# ita = italiano
# jpn = japonês
# cmn = chinês (mandarim)
# rus = russo
# ara = árabe
