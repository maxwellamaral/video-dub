"""
Serviço de Análise de Emoções em Áudio com SenseVoice.

Este módulo utiliza o modelo SenseVoiceSmall da FunAudioLLM para detectar emoções
em segmentos de áudio transcritos. As emoções detectadas são integradas às legendas
e usadas para controlar a expressividade do TTS Qwen3.

SenseVoiceSmall Features:
- Detecção de emoções: Angry, Happy, Sad, Neutral, etc.
- Suporte multilíngue
- Alta precisão em Speech Emotion Recognition (SER)

Referências:
- https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- https://github.com/FunAudioLLM/SenseVoice
"""

import os
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from src.config import DEVICE

# Mapeamento de emoções detectadas pelo SenseVoice para descrições em português
# utilizadas nas instruções do Qwen3-TTS
EMOTION_MAP_PT = {
    "angry": "zangado",
    "happy": "feliz",
    "sad": "triste",
    "neutral": "neutro",
    "fearful": "amedrontado",
    "disgusted": "enojado",
    "surprised": "surpreso"
}

# Mapeamento de emoções para instruções detalhadas do Qwen3-TTS
# Estas instruções orientam o modelo TTS a sintetizar com a emoção apropriada
EMOTION_INSTRUCTIONS = {
    "angry": "Fale com tom zangado, voz elevada e ritmo acelerado, demonstrando irritação e frustração",
    "happy": "Fale com tom alegre e entusiasmado, voz animada e expressiva, transmitindo felicidade",
    "sad": "Fale com tom triste e melancólico, voz baixa e lenta, demonstrando tristeza profunda",
    "neutral": "Fale com tom neutro e equilibrado, voz clara e natural, sem ênfase emocional",
    "fearful": "Fale com tom amedrontado e hesitante, voz trêmula e acelerada, demonstrando medo",
    "disgusted": "Fale com tom de desgosto e repulsa, voz áspera, demonstrando aversão",
    "surprised": "Fale com tom surpreso e espantado, voz animada com inflexões súbitas, demonstrando choque"
}


class EmotionAnalyzer:
    """
    Analisador de Emoções em Áudio usando SenseVoiceSmall.
    
    Detecta emoções em segmentos de áudio e fornece tags e instruções
    para integração com legendas e síntese TTS expressiva.
    """
    
    def __init__(self, modelo="FunAudioLLM/SenseVoiceSmall", log_callback=None):
        """
        Inicializa o analisador de emoções.
        
        Args:
            modelo (str): ID do modelo no Hugging Face ou caminho local.
            log_callback (callable, optional): Função para logar mensagens.
        """
        self.modelo_nome = modelo
        self.log_callback = log_callback
        self.model = None
        self.processor = None
        
        self._carregar_modelo()
    
    def _log(self, msg):
        """Helper para logging."""
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)
    
    def _carregar_modelo(self):
        """
        Carrega o modelo SenseVoice e o processador.
        
        O modelo é carregado em modo offline quando OFFLINE_MODE=True
        e utiliza GPU se disponível para melhor desempenho.
        """
        try:
            self._log(f"   🎭 Carregando SenseVoice: {self.modelo_nome}")
            
            from src.config import OFFLINE_MODE
            
            # Configurações para carregamento
            load_kwargs = {
                "torch_dtype": torch.float16 if "cuda" in DEVICE else torch.float32,
                "low_cpu_mem_usage": True,
                "use_safetensors": True
            }
            
            if OFFLINE_MODE:
                load_kwargs["local_files_only"] = True
                self._log("   Modo offline ativado para SenseVoice")
            
            # Carregar processador (tokenizer + feature extractor)
            self.processor = AutoProcessor.from_pretrained(
                self.modelo_nome,
                **load_kwargs
            )
            
            # Carregar modelo
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.modelo_nome,
                **load_kwargs
            ).to(DEVICE)
            
            self.model.eval()  # Modo de inferência
            
            self._log(f"   ✓ SenseVoice carregado em {DEVICE}")
            
        except Exception as e:
            self._log(f"   ✗ Erro ao carregar SenseVoice: {e}")
            raise
    
    def analisar_audio(self, caminho_audio, segmentos=None):
        """
        Analisa emoções em um arquivo de áudio completo ou em segmentos específicos.
        
        Args:
            caminho_audio (str): Caminho do arquivo de áudio (.wav).
            segmentos (list, optional): Lista de segmentos com 'start', 'end', 'text'.
                                       Se None, analisa o áudio completo.
        
        Returns:
            list ou dict: Se segmentos fornecidos, retorna lista de segmentos enriquecidos
                         com 'emotion' e 'emotion_instruction'. Caso contrário, retorna
                         dict com emoção detectada no áudio completo.
        """
        if not os.path.exists(caminho_audio):
            self._log(f"   ✗ Áudio não encontrado: {caminho_audio}")
            return segmentos if segmentos else {"emotion": "neutral", "confidence": 0.0}
        
        if segmentos:
            return self._analisar_segmentos(caminho_audio, segmentos)
        else:
            return self._analisar_audio_completo(caminho_audio)
    
    def _analisar_audio_completo(self, caminho_audio):
        """
        Analisa emoção do áudio completo.
        
        Args:
            caminho_audio (str): Caminho do arquivo de áudio.
        
        Returns:
            dict: Dicionário com 'emotion', 'confidence', 'emotion_pt', 'instruction'.
        """
        self._log(f"   🎭 Analisando emoção do áudio completo...")
        
        try:
            # Carregar áudio
            import soundfile as sf
            audio_data, sample_rate = sf.read(caminho_audio)
            
            # Converter para mono se estéreo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Processar áudio
            inputs = self.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Inferência
            with torch.no_grad():
                outputs = self.model.generate(**inputs, return_dict_in_generate=True)
            
            # Decodificar resultado
            # SenseVoice retorna texto transcrito com tags de emoção no formato <|EMOTION|>
            transcription = self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=False
            )[0]
            
            # Extrair emoção das tags especiais
            emotion = self._extrair_emocao_de_tag(transcription)
            
            return {
                "emotion": emotion,
                "emotion_pt": EMOTION_MAP_PT.get(emotion, "neutro"),
                "instruction": EMOTION_INSTRUCTIONS.get(emotion, ""),
                "confidence": 1.0  # SenseVoice não retorna score de confiança diretamente
            }
            
        except Exception as e:
            self._log(f"   ⚠️ Erro ao analisar emoção: {e}")
            return {
                "emotion": "neutral",
                "emotion_pt": "neutro",
                "instruction": EMOTION_INSTRUCTIONS["neutral"],
                "confidence": 0.0
            }
    
    def _analisar_segmentos(self, caminho_audio, segmentos):
        """
        Analisa emoções em múltiplos segmentos de áudio.
        
        Para cada segmento, extrai o trecho de áudio correspondente e detecta a emoção.
        Enriquece os segmentos com informações emocionais para uso em legendas e TTS.
        
        Args:
            caminho_audio (str): Caminho do arquivo de áudio completo.
            segmentos (list): Lista de dicts com 'start', 'end', 'text'.
        
        Returns:
            list: Segmentos enriquecidos com campos adicionais:
                  - 'emotion': código da emoção (ex: 'happy', 'sad')
                  - 'emotion_pt': emoção em português (ex: 'feliz', 'triste')
                  - 'emotion_instruction': instrução para Qwen3-TTS
        """
        self._log(f"   🎭 Analisando emoções de {len(segmentos)} segmentos...")
        
        try:
            import soundfile as sf
            
            # Carregar áudio completo uma vez
            audio_completo, sample_rate = sf.read(caminho_audio)
            
            # Converter para mono se necessário
            if len(audio_completo.shape) > 1:
                audio_completo = audio_completo.mean(axis=1)
            
            segmentos_enriquecidos = []
            
            for i, seg in enumerate(segmentos):
                # Logar progresso a cada 10 segmentos
                if (i + 1) % 10 == 0:
                    self._log(f"   ... Analisando emoção {i+1}/{len(segmentos)}")
                
                # Extrair trecho de áudio do segmento
                inicio_sample = int(seg["start"] * sample_rate)
                fim_sample = int(seg["end"] * sample_rate)
                
                # Validar limites
                inicio_sample = max(0, inicio_sample)
                fim_sample = min(len(audio_completo), fim_sample)
                
                if inicio_sample >= fim_sample:
                    # Segmento inválido - usar emoção neutra
                    seg_copy = seg.copy()
                    seg_copy["emotion"] = "neutral"
                    seg_copy["emotion_pt"] = "neutro"
                    seg_copy["emotion_instruction"] = EMOTION_INSTRUCTIONS["neutral"]
                    segmentos_enriquecidos.append(seg_copy)
                    continue
                
                audio_segmento = audio_completo[inicio_sample:fim_sample]
                
                # Detectar emoção no segmento
                emotion_data = self._detectar_emocao_trecho(audio_segmento, sample_rate)
                
                # Enriquecer segmento com dados de emoção
                seg_enriquecido = seg.copy()
                seg_enriquecido["emotion"] = emotion_data["emotion"]
                seg_enriquecido["emotion_pt"] = emotion_data["emotion_pt"]
                seg_enriquecido["emotion_instruction"] = emotion_data["instruction"]
                
                segmentos_enriquecidos.append(seg_enriquecido)
            
            self._log(f"   ✓ Análise de emoções concluída: {len(segmentos_enriquecidos)} segmentos")
            
            return segmentos_enriquecidos
            
        except Exception as e:
            self._log(f"   ⚠️ Erro ao analisar segmentos: {e}")
            # Em caso de erro, retornar segmentos originais com emoção neutra
            return [
                {**seg, "emotion": "neutral", "emotion_pt": "neutro", 
                 "emotion_instruction": EMOTION_INSTRUCTIONS["neutral"]}
                for seg in segmentos
            ]
    
    def _detectar_emocao_trecho(self, audio_data, sample_rate):
        """
        Detecta emoção em um trecho de áudio (numpy array).
        
        Args:
            audio_data (np.ndarray): Array numpy com dados de áudio.
            sample_rate (int): Taxa de amostragem do áudio.
        
        Returns:
            dict: Dicionário com 'emotion', 'emotion_pt', 'instruction'.
        """
        try:
            # Processar áudio
            inputs = self.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Inferência
            with torch.no_grad():
                outputs = self.model.generate(**inputs, return_dict_in_generate=True)
            
            # Decodificar
            transcription = self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=False
            )[0]
            
            # Extrair emoção
            emotion = self._extrair_emocao_de_tag(transcription)
            
            return {
                "emotion": emotion,
                "emotion_pt": EMOTION_MAP_PT.get(emotion, "neutro"),
                "instruction": EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS["neutral"])
            }
            
        except Exception as e:
            self._log(f"   ⚠️ Erro ao detectar emoção em trecho: {e}")
            return {
                "emotion": "neutral",
                "emotion_pt": "neutro",
                "instruction": EMOTION_INSTRUCTIONS["neutral"]
            }
    
    def _extrair_emocao_de_tag(self, transcription):
        """
        Extrai a emoção das tags especiais do SenseVoice.
        
        SenseVoice retorna transcrições com tags especiais no formato:
        <|emotion|> onde emotion pode ser: HAPPY, SAD, ANGRY, NEUTRAL, etc.
        
        Args:
            transcription (str): Texto transcrito com tags especiais.
        
        Returns:
            str: Código da emoção em minúsculas (ex: 'happy', 'sad', 'neutral').
        """
        import re
        
        # Padrão para encontrar tags de emoção: <|EMOTION|>
        pattern = r'<\|([A-Z]+)\|>'
        matches = re.findall(pattern, transcription.upper())
        
        if matches:
            # Pegar primeira emoção detectada
            emotion_tag = matches[0].lower()
            
            # Mapear para emoções suportadas
            emotion_mapping = {
                "happy": "happy",
                "sad": "sad",
                "angry": "angry",
                "neutral": "neutral",
                "fear": "fearful",
                "fearful": "fearful",
                "disgust": "disgusted",
                "disgusted": "disgusted",
                "surprise": "surprised",
                "surprised": "surprised"
            }
            
            return emotion_mapping.get(emotion_tag, "neutral")
        
        # Se não encontrar tag, assumir neutro
        return "neutral"
    
    def formatar_legenda_com_emocao(self, segmentos_com_emocao):
        """
        Formata segmentos enriquecidos para exibição em legendas.
        
        Adiciona tags de emoção às legendas no formato:
        [EMOÇÃO] Texto da legenda
        
        Args:
            segmentos_com_emocao (list): Segmentos com campo 'emotion_pt'.
        
        Returns:
            list: Segmentos com campo 'text' formatado com tag de emoção.
        """
        segmentos_formatados = []
        
        for seg in segmentos_com_emocao:
            seg_copy = seg.copy()
            
            # Adicionar tag de emoção ao texto apenas se não for neutro
            if seg.get("emotion", "neutral") != "neutral":
                emotion_tag = f"[{seg['emotion_pt'].upper()}]"
                seg_copy["text"] = f"{emotion_tag} {seg['text']}"
            
            segmentos_formatados.append(seg_copy)
        
        return segmentos_formatados
