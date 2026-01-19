
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
    - 'coqui': Coqui XTTS v2 - Qualidade alta, suporte a clonagem de voz.
    """
    def __init__(self, motor="mms", idioma="por", ref_wav=None, log_callback=None):
        """
        Inicializa o motor TTS.

        Args:
            motor (str): 'mms' ou 'coqui'.
            idioma (str): Código do idioma (ex: 'por', 'por_Latn').
            ref_wav (str, optional): Caminho para áudio de referência (apenas Coqui).
            log_callback (callable, optional): Função para logar mensagens.
        """
        self.motor = motor
        self.idioma = idioma
        self.ref_wav = ref_wav
        self.log_callback = log_callback
        self.config = {}
        self.sample_rate = 24000 # default fallback
        
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
                
            elif self.motor == "coqui":
                self._log("   Carregando Coqui XTTS v2...")
                os.environ["COQUI_TOS_AGREED"] = "1"
                from TTS.api import TTS
                
                # Patch para torch.load weights_only (PyTorch 2.6+ fix)
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load
                
                self.config["tts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
                torch.load = original_load # Restore
                
                self.sample_rate = 24000
                self._log("   ✓ Coqui Loaded.")
                
        except Exception as e:
            self._log(f"✗ Erro ao inicializar TTS {self.motor}: {e}")
            raise e

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
                    
        elif self.motor == "coqui":
            tts = self.config["tts"]
            lang = "pt" if self.idioma == "por_Latn" or self.idioma == "por" else "en"
            
            for i, texto in enumerate(textos):
                if (i+1) % 5 == 0: self._log(f"   ... Sintetizando {i+1}/{len(textos)}")
                try:
                    wav = tts.tts(text=texto, speaker_wav=self.ref_wav, language=lang)
                    resultados.append((np.array(wav), self.sample_rate))
                except Exception as e:
                    self._log(f"   ⚠️ Erro Coqui no segmento {i}: {e}")
                    resultados.append((None, None))
                    
        return resultados
