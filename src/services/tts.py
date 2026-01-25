
import os
import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
from src.config import DEVICE

class TTSEngine:
    """
    Motor unificado de Síntese de Voz (Text-to-Speech).

    Suporta múltiplos backends:
    - 'mms': Meta Massively Multilingual Speech (Facebook) - Rápido, offline.
    - 'qwen3': Qwen3-TTS CustomVoice - Alta qualidade, latência ultra-baixa, controle expressivo.
    """
    def __init__(self, motor="mms", idioma="por", ref_wav=None, log_callback=None,
                 qwen3_mode="custom", qwen3_speaker="vivian", qwen3_instruct=""):
        """
        Inicializa o motor TTS.

        Args:
            motor (str): 'mms' ou 'qwen3'.
            idioma (str): Código do idioma (ex: 'por', 'por_Latn').
            ref_wav (str, optional): Caminho para áudio de referência (Qwen3-Clone).
            log_callback (callable, optional): Função para logar mensagens.
            qwen3_mode (str): Modalidade Qwen3: 'custom', 'design', ou 'clone'.
            qwen3_speaker (str): Speaker para modo CustomVoice (ex: 'Vivian', 'Ryan').
            qwen3_instruct (str): Instrução de controle de voz (CustomVoice/VoiceDesign).
        """
        self.motor = motor
        self.idioma = idioma
        self.ref_wav = ref_wav
        self.log_callback = log_callback
        self.config = {}
        
        # Parâmetros Qwen3-TTS
        self.qwen3_mode = qwen3_mode
        self.qwen3_speaker = qwen3_speaker
        self.qwen3_instruct = qwen3_instruct
        self.sample_rate = 24000 # default fallback
        self.speaker = None  # Para Qwen3-TTS
        
        self._carregar_modelo()
        
    def _log(self, msg):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

    def _carregar_modelo(self):
        try:
            if self.motor == "mms":
                modelo_nome = f"facebook/mms-tts-{self.idioma}"
                self._log(f"   Carregando MMS-TTS: {modelo_nome}")
                
                self.config["tokenizer"] = AutoTokenizer.from_pretrained(modelo_nome)
                self.config["model"] = VitsModel.from_pretrained(modelo_nome).to(DEVICE)
                
                self.sample_rate = self.config["model"].config.sampling_rate
                

            elif self.motor == "qwen3":
                self._log(f"   Carregando Qwen3-TTS ({self.qwen3_mode} mode)...")
                from src.config import QWEN3_DEFAULT_SPEAKER
                
                try:
                    from qwen_tts import Qwen3TTSModel
                    
                    # Selecionar modelo baseado na modalidade
                    model_map = {
                        "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                    }
                    
                    model_name = model_map.get(self.qwen3_mode, model_map["custom"])
                    self._log(f"   Modelo: {model_name}")
                    
                    # Configuração para modo offline
                    load_kwargs = {
                        "device_map": DEVICE,
                        "dtype": torch.bfloat16,
                        "local_files_only": True,  # Modo offline
                        "trust_remote_code": True   # Necessário para modelos custom
                    }
                    
                    # Tentar com FlashAttention 2 primeiro
                    try:
                        self._log("   Tentando com FlashAttention 2...")
                        self.config["model"] = Qwen3TTSModel.from_pretrained(
                            model_name,
                            attn_implementation="flash_attention_2",
                            **load_kwargs
                        )
                        self._log("   ✓ FlashAttention 2 ativado")
                    except Exception as fa_error:
                        self._log(f"   ⚠️ FlashAttention 2 não disponível: {fa_error}")
                        self._log("   Usando implementação padrão...")
                        self.config["model"] = Qwen3TTSModel.from_pretrained(
                            model_name,
                            **load_kwargs
                        )
                    
                    # Todos os modelos Qwen3-TTS-12Hz geram áudio a 12kHz
                    self.sample_rate = 12000
                    self.speaker = self.qwen3_speaker if self.qwen3_mode == "custom" else None
                    
                    # Mapear idioma para formato Qwen3
                    self.qwen3_language = self._mapear_idioma_qwen3()
                    
                    mode_desc = {
                        "custom": f"CustomVoice (speaker: {self.speaker})",
                        "design": "VoiceDesign (free-form)",
                        "clone": "Clone (voice cloning)"
                    }
                    
                    self._log(f"   ✓ Qwen3-TTS carregado: {mode_desc[self.qwen3_mode]}, lang: {self.qwen3_language}")
                    
                except ImportError:
                    self._log("✗ Pacote 'qwen-tts' não instalado. Execute: uv add qwen-tts")
                    raise
                
                
        except Exception as e:
            self._log(f"✗ Erro ao inicializar TTS {self.motor}: {e}")
            raise e
    
    def _mapear_idioma_qwen3(self):
        """Mapeia código de idioma para formato Qwen3-TTS."""
        mapeamento = {
            "por": "Portuguese",
            "por_Latn": "Portuguese",
            "eng": "English",
            "eng_Latn": "English",
            "spa": "Spanish",
            "spa_Latn": "Spanish",
            "fra": "French",
            "fra_Latn": "French",
            "deu": "German",
            "deu_Latn": "German",
            "ita": "Italian",
            "ita_Latn": "Italian",
            "jpn": "Japanese",
            "jpn_Jpan": "Japanese",
            "kor": "Korean",
            "kor_Hang": "Korean",
            "rus": "Russian",
            "rus_Cyrl": "Russian",
            "cmn": "Chinese",
            "zho": "Chinese",
        }
        return mapeamento.get(self.idioma, "Auto")

    def sintetizar_batch(self, textos):
        """
        Sintetiza uma lista de textos em áudio.

        Para MMS, tenta processar em lote (embora a implementação atual seja iterativa
        para evitar OOM, a interface permite otimização futura).
        
        Args:
            textos (list): Lista de strings para sintetizar.

        Returns:
            list: Lista de tuplas (audio_numpy_array, sample_rate).
                  Retorna (None, None) em caso de falha no segmento.
        """
        self._log(f"   🔊 Sintetizando {len(textos)} segmentos ({self.motor})...")
        resultados = []
        
        if self.motor == "mms":
            model = self.config["model"]
            tokenizer = self.config["tokenizer"]
            
            with torch.no_grad():
                for i, texto in enumerate(textos):
                    # Logs de progresso
                    if (i+1) % 5 == 0: self._log(f"   ... Sintetizando {i+1}/{len(textos)}")

                    clean = "".join([c for c in texto if c.isalnum() or c in " ,.?!"])
                    if not clean.strip():
                        resultados.append((None, None))
                        continue
                        
                    inputs = tokenizer(clean, return_tensors="pt").to(DEVICE)
                    output = model(**inputs).waveform
                    audio = output.cpu().numpy().squeeze()
                    resultados.append((audio, self.sample_rate))
                    

        elif self.motor == "qwen3":
            model = self.config["model"]
            
            for i, texto in enumerate(textos):
                if (i+1) % 5 == 0: self._log(f"   ... Sintetizando {i+1}/{len(textos)}")
                
                try:
                    # Limpeza básica
                    clean = texto.strip()
                    if not clean:
                        resultados.append((None, None))
                        continue
                    
                    # Síntese baseada na modalidade
                    if self.qwen3_mode == "custom":
                        # CustomVoice: usa speaker pré-definido + instrução opcional
                        wavs, sr = model.generate_custom_voice(
                            text=clean,
                            language=self.qwen3_language,
                            speaker=self.speaker,
                            instruct=self.qwen3_instruct if self.qwen3_instruct else ""
                        )
                        
                    elif self.qwen3_mode == "design":
                        # VoiceDesign: cria voz baseada em descrição em linguagem natural
                        instruct = self.qwen3_instruct or "Voz clara e natural, tom neutro e profissional"
                        wavs, sr = model.generate_voice_design(
                            text=clean,
                            language=self.qwen3_language,
                            instruct=instruct
                        )
                        
                    elif self.qwen3_mode == "clone":
                        # Clone: clona voz a partir de áudio de referência
                        if not self.ref_wav or not os.path.exists(self.ref_wav):
                            self._log(f"   ⚠️ Áudio de referência não encontrado: {self.ref_wav}")
                            resultados.append((None, None))
                            continue
                        
                        wavs, sr = model.generate_voice_clone(
                            text=clean,
                            language=self.qwen3_language,
                            ref_audio=self.ref_wav,
                            ref_text="",  # Opcional: transcrição do áudio de referência
                            x_vector_only_mode=True  # Usar apenas embedding de speaker
                        )
                    else:
                        self._log(f"   ❌ Modo Qwen3 desconhecido: {self.qwen3_mode}")
                        resultados.append((None, None))
                        continue
                    
                    # Extrair primeiro áudio do batch
                    if wavs and len(wavs) > 0:
                        audio_data = wavs[0]
                        resultados.append((audio_data, sr))
                    else:
                        resultados.append((None, None))
                        
                except Exception as e:
                    self._log(f"   ⚠️ Erro Qwen3 no segmento {i}: {e}")
                    resultados.append((None, None))
                    
        return resultados
