
import torch
from transformers import pipeline
from src.config import DEVICE

def traduzir_segmentos(segmentos, idioma_origem, idioma_destino):
    """
    Traduz lista de segmentos preservando timestamps usando NLLB.
    """
    print(f"\n🌐 Traduzindo de {idioma_origem} para {idioma_destino}...")
    
    try:
        pipe = pipeline(
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
            if not texto: continue
            
            if (i+1) % 10 == 0: print(f"   Segmento {i+1}/{total}")
            
            try:
                # Max length seguro para legendas
                res = pipe(texto, max_length=512)
                texto_trad = res[0]["translation_text"]
                
                segmentos_traduzidos.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": texto_trad
                })
            except Exception as e:
                print(f"   ⚠️  Erro segmento {i+1}: {e}")
                # Fallback: original
                segmentos_traduzidos.append(seg)
                
        return segmentos_traduzidos
        
    except Exception as e:
        print(f"✗ Erro ao carregar modelo de tradução: {e}")
        return segmentos # Devolve original se falhar tudo
