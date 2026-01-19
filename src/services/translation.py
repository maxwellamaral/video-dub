
import torch
from transformers import pipeline
from src.config import DEVICE

def traduzir_segmentos(segmentos, idioma_origem, idioma_destino, log_callback=None):
    """
    Traduz uma lista de segmentos de texto preservando os timestamps originais.

    Utiliza o modelo NLLB (No Language Left Behind) da Meta (Facebook) para
    tradução neural de alta qualidade.

    Args:
        segmentos (list): Lista de dicts {'start', 'end', 'text'}.
        idioma_origem (str): Código NLLB do idioma fonte (ex: 'eng_Latn').
        idioma_destino (str): Código NLLB do idioma alvo (ex: 'por_Latn').
        log_callback (callable, optional): Função para logar mensagens.

    Returns:
        list: Nova lista de segmentos com a chave 'text' traduzida.
    """
    msg = f"\n🌐 Traduzindo de {idioma_origem} para {idioma_destino}..."
    if log_callback: log_callback(msg)
    else: print(msg)
    
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
        msg_total = f"   Traduzindo {total} segmentos..."
        if log_callback: log_callback(msg_total)
        else: print(msg_total)
        
        for i, seg in enumerate(segmentos):
            texto = seg["text"].strip()
            if not texto: continue
            
            # Log periódico
            if (i+1) % 5 == 0 or i == total-1:
                prog = f"   ... Traduzindo segmento {i+1}/{total}"
                if log_callback: log_callback(prog)
                else: print(prog)
            
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
                err = f"   ⚠️  Erro no segmento {i+1}: {e}"
                if log_callback: log_callback(err)
                else: print(err)
                # Fallback: original
                segmentos_traduzidos.append(seg)
                
        return segmentos_traduzidos
        
    except Exception as e:
        err_fatal = f"✗ Erro ao carregar modelo de tradução: {e}"
        if log_callback: log_callback(err_fatal)
        else: print(err_fatal)
        return segmentos # Devolve original se falhar tudo
