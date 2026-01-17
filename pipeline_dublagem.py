# ============================================================================
# PIPELINE DE DUBLAGEM DE VÍDEOS COM HUGGING FACE
# Transcrição → Tradução → Síntese de Voz → Remontagem
# Otimizado para GPU NVIDIA
# ============================================================================

import os
import torch
import torchaudio
import subprocess
from pathlib import Path
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import VitsModel, AutoTokenizer as TTSTokenizer
import numpy as np

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
VIDEO_ENTRADA = "video_entrada.mp4"      # Seu vídeo
AUDIO_EXTRAIDO = "audio_extraido.wav"
AUDIO_TRADUZIDO = "audio_traduzido.wav"
VIDEO_SAIDA = "video_dublado.mp4"
LEGENDA_ORIGINAL = "legenda_original.srt"   # Legenda original com timestamps
LEGENDA_TRADUZIDA = "legenda_traduzida.srt" # Legenda traduzida com timestamps

# ============================================================================
# ETAPA 1: EXTRAÇÃO DE ÁUDIO DO VÍDEO (usando ffmpeg)
# ============================================================================

def extrair_audio(caminho_video, caminho_audio_saida):
    """
    Extrai áudio do vídeo usando ffmpeg.
    Certifique-se de ter ffmpeg instalado: apt-get install ffmpeg (Linux/WSL)
    """
    print(f"\n📹 Extraindo áudio de: {caminho_video}")
    try:
        cmd = [
            "ffmpeg", "-i", caminho_video,
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
            "ffmpeg", "-i", caminho_video_orig,
            "-i", caminho_audio_novo,
            "-c:v", "copy",           # Copiar vídeo sem recodificar (mais rápido)
            "-c:a", "aac",            # Codec de áudio
            "-map", "0:v:0",          # Mapear vídeo do arquivo 0
            "-map", "1:a:0",          # Mapear áudio do arquivo 1 (novo)
            "-shortest",              # Cortar ao comprimento mais curto
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
    idioma_voz="por"
):
    """
    Executa a pipeline completa de dublagem.
    """
    print("=" * 70)
    print("INICIANDO PIPELINE DE DUBLAGEM")
    print("=" * 70)
    
    # 1. Extrair áudio
    if not extrair_audio(caminho_video, AUDIO_EXTRAIDO):
        print("✗ Falha na extração de áudio. Abortando.")
        return False
    
    # 2. Transcrever (retorna segmentos)
    segmentos_originais = transcrever_audio(AUDIO_EXTRAIDO)
    if not segmentos_originais:
        print("✗ Falha na transcrição. Abortando.")
        return False
    
    # 2.1. Salvar legenda original (SRT)
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
    
    # 3.1. Salvar legenda traduzida (SRT)
    try:
        conteudo_srt_trad = segmentos_para_srt(segmentos_traduzidos)
        with open(LEGENDA_TRADUZIDA, "w", encoding="utf-8") as f:
            f.write(conteudo_srt_trad)
        print(f"✓ Legenda traduzida salva em: {LEGENDA_TRADUZIDA}")
    except Exception as e:
        print(f"⚠️  Aviso: Não foi possível salvar legenda traduzida: {e}")
    
    # Preparar texto para TTS (juntar segmentos)
    texto_para_tts = segmentos_para_texto(segmentos_traduzidos)
    
    # 4. Sintetizar voz
    audio_novo, sample_rate = sintetizar_voz(texto_para_tts, idioma_voz)
    if audio_novo is None:
        print("✗ Falha na síntese de voz. Abortando.")
        return False
    
    # 5. Salvar novo áudio
    if not salvar_audio(audio_novo, sample_rate, AUDIO_TRADUZIDO):
        print("✗ Falha ao salvar áudio. Abortando.")
        return False
    
    # 6. Remontar vídeo
    if not remontar_video(caminho_video, AUDIO_TRADUZIDO, VIDEO_SAIDA):
        print("✗ Falha na remontagem. Abortando.")
        return False
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETADA COM SUCESSO!")
    print(f"✓ Vídeo dublado salvo em: {VIDEO_SAIDA}")
    print(f"✓ Legenda original salva em: {LEGENDA_ORIGINAL}")
    print(f"✓ Legenda traduzida salva em: {LEGENDA_TRADUZIDA}")
    print("=" * 70)
    
    return True

# ============================================================================
# MAIN: EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Certifique-se de que seu vídeo existe
    if not os.path.exists(VIDEO_ENTRADA):
        print(f"✗ Arquivo não encontrado: {VIDEO_ENTRADA}")
        print("  Coloque seu vídeo no mesmo diretório e renomeie para 'video_entrada.mp4'")
    else:
        # Executar pipeline completa
        sucesso = executar_pipeline_completa(
            caminho_video=VIDEO_ENTRADA,
            idioma_origem="eng_Latn",   # Inglês (NLLB)
            idioma_destino="por_Latn",  # Português (NLLB)
            idioma_voz="por"             # Português (MMS-TTS)
        )
        
        if sucesso:
            print("\n💡 Dicas:")
            print("  • Para vídeos mais longos, considere usar modelo 'tiny' no Whisper")
            print("  • Se ficar sem memória, reduza para float16 ou use modelos menores")
            print("  • Edite os códigos de idioma conforme necessário no topo do arquivo")

# ============================================================================
# REFERÊNCIAS DE IDIOMAS (NLLB e MMS-TTS)
# ============================================================================

# Códigos NLLB (amostra):
# en_XX = inglês
# pt_BR = português brasileiro
# pt_PT = português europeu
# es_ES = espanhol
# fr_XX = francês
# de_DE = alemão
# it_IT = italiano
# ja_XX = japonês
# zh_Hans = chinês simplificado
# ru_RU = russo
# ar_AR = árabe

# Idiomas MMS-TTS (texto-para-fala):
# pt = português
# en = inglês
# es = espanhol
# fr = francês
# de = alemão
# it = italiano
# ja = japonês
# zh = chinês
# ru = russo
# ar = árabe
