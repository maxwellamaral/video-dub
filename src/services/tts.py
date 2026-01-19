
import os
import torch
import numpy as np
from transformers import VitsModel, AutoTokenizer
from src.config import DEVICE

class TTSEngine:
    def __init__(self, motor="mms", idioma="por", ref_wav=None):
        self.motor = motor
        self.idioma = idioma
        self.ref_wav = ref_wav
        self.config = {}
        self.sample_rate = 24000 # default fallback
        
        self._carregar_modelo()
        
    def _carregar_modelo(self):
        try:
            if self.motor == "mms":
                modelo_nome = f"facebook/mms-tts-{self.idioma}"
                print(f"   Carregando MMS-TTS: {modelo_nome}")
                self.config["tokenizer"] = AutoTokenizer.from_pretrained(modelo_nome)
                self.config["model"] = VitsModel.from_pretrained(modelo_nome).to(DEVICE)
                self.sample_rate = self.config["model"].config.sampling_rate
                
            elif self.motor == "coqui":
                print("   Carregando Coqui XTTS v2...")
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
                print("   ✓ Coqui Loaded.")
                
        except Exception as e:
            print(f"✗ Erro ao inicializar TTS {self.motor}: {e}")
            raise e

    def sintetizar_batch(self, textos):
        """
        Sintetiza uma lista de textos.
        Retorna lista de tuplas (audio_numpy, sample_rate).
        """
        print(f"   🔊 Sintetizando {len(textos)} segmentos ({self.motor})...")
        resultados = []
        
        if self.motor == "mms":
            model = self.config["model"]
            tokenizer = self.config["tokenizer"]
            
            with torch.no_grad():
                for texto in textos:
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
            
            for texto in textos:
                try:
                    wav = tts.tts(text=texto, speaker_wav=self.ref_wav, language=lang)
                    resultados.append((np.array(wav), self.sample_rate))
                except Exception as e:
                    print(f"   ⚠️ Erro Coqui no segmento: {e}")
                    resultados.append((None, None))
                    
        return resultados
