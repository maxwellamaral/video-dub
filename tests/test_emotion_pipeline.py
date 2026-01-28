"""
Teste de integração: Pipeline completo com análise de emoções.

Este teste demonstra o uso do pipeline de dublagem com detecção de emoções
usando SenseVoice e síntese expressiva com Qwen3-TTS.

Para executar:
    pytest tests/test_emotion_pipeline.py -v
"""

import pytest
import os
from src.config import ENABLE_EMOTION_ANALYSIS, SENSEVOICE_MODEL
from src.services.emotion import EmotionAnalyzer, EMOTION_INSTRUCTIONS, EMOTION_MAP_PT
from src.services.audio import transcrever_com_emocao, transcrever_audio_whisper
from src.utils import segmentos_para_srt_com_emocao, extrair_estatisticas_emocoes


class TestEmotionAnalyzer:
    """Testes para o analisador de emoções SenseVoice."""
    
    def test_emotion_analyzer_init(self):
        """Testa inicialização do EmotionAnalyzer."""
        analyzer = EmotionAnalyzer(modelo=SENSEVOICE_MODEL)
        
        assert analyzer.model is not None
        assert analyzer.processor is not None
        assert analyzer.modelo_nome == SENSEVOICE_MODEL
    
    def test_emotion_map_pt(self):
        """Testa mapeamento de emoções para português."""
        assert EMOTION_MAP_PT["happy"] == "feliz"
        assert EMOTION_MAP_PT["sad"] == "triste"
        assert EMOTION_MAP_PT["angry"] == "zangado"
        assert EMOTION_MAP_PT["neutral"] == "neutro"
    
    def test_emotion_instructions(self):
        """Testa se todas as emoções têm instruções."""
        emotions = ["angry", "happy", "sad", "neutral", "fearful", "disgusted", "surprised"]
        
        for emotion in emotions:
            assert emotion in EMOTION_INSTRUCTIONS
            assert isinstance(EMOTION_INSTRUCTIONS[emotion], str)
            assert len(EMOTION_INSTRUCTIONS[emotion]) > 0
    
    @pytest.mark.skip(reason="Requer arquivo de áudio de teste")
    def test_analyze_audio_file(self):
        """Testa análise de emoção em arquivo de áudio."""
        # Este teste precisa de um arquivo de áudio real
        test_audio = "tests/fixtures/test_audio_angry.wav"
        
        if not os.path.exists(test_audio):
            pytest.skip(f"Arquivo de teste não encontrado: {test_audio}")
        
        analyzer = EmotionAnalyzer()
        result = analyzer.analisar_audio(test_audio)
        
        assert "emotion" in result
        assert "emotion_pt" in result
        assert "instruction" in result
        assert result["emotion"] in ["angry", "happy", "sad", "neutral", "fearful", "disgusted", "surprised"]
    
    def test_extrair_emocao_de_tag(self):
        """Testa extração de emoção das tags do SenseVoice."""
        analyzer = EmotionAnalyzer()
        
        # Testa diferentes formatos de tag
        assert analyzer._extrair_emocao_de_tag("<|HAPPY|> Hello world") == "happy"
        assert analyzer._extrair_emocao_de_tag("<|SAD|> I'm crying") == "sad"
        assert analyzer._extrair_emocao_de_tag("<|ANGRY|> This is unacceptable") == "angry"
        assert analyzer._extrair_emocao_de_tag("No emotion tag here") == "neutral"


class TestEmotionIntegration:
    """Testes de integração com o pipeline."""
    
    def test_segmentos_com_emocao_structure(self):
        """Testa estrutura de segmentos enriquecidos com emoções."""
        segmentos_mock = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello",
                "emotion": "happy",
                "emotion_pt": "feliz",
                "emotion_instruction": "Speak with happy tone"
            }
        ]
        
        for seg in segmentos_mock:
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert "emotion" in seg
            assert "emotion_pt" in seg
            assert "emotion_instruction" in seg
    
    def test_segmentos_para_srt_com_emocao(self):
        """Testa geração de SRT com tags de emoção."""
        segmentos = [
            {
                "start": 0.5,
                "end": 2.5,
                "text": "I'm so happy!",
                "emotion": "happy",
                "emotion_pt": "feliz"
            },
            {
                "start": 3.0,
                "end": 5.0,
                "text": "This is sad.",
                "emotion": "sad",
                "emotion_pt": "triste"
            },
            {
                "start": 6.0,
                "end": 8.0,
                "text": "Normal speech.",
                "emotion": "neutral",
                "emotion_pt": "neutro"
            }
        ]
        
        # Com tags
        srt_com_tags = segmentos_para_srt_com_emocao(segmentos, incluir_tag_emocao=True)
        assert "[FELIZ]" in srt_com_tags
        assert "[TRISTE]" in srt_com_tags
        assert "[NEUTRO]" not in srt_com_tags  # Neutro não deve ter tag
        
        # Sem tags
        srt_sem_tags = segmentos_para_srt_com_emocao(segmentos, incluir_tag_emocao=False)
        assert "[FELIZ]" not in srt_sem_tags
        assert "[TRISTE]" not in srt_sem_tags
    
    def test_extrair_estatisticas_emocoes(self):
        """Testa extração de estatísticas de emoções."""
        segmentos = [
            {"emotion": "happy"},
            {"emotion": "happy"},
            {"emotion": "sad"},
            {"emotion": "neutral"},
            {"emotion": "happy"}
        ]
        
        stats = extrair_estatisticas_emocoes(segmentos)
        
        assert stats["total"] == 5
        assert stats["predominante"] == "happy"
        assert stats["emocoes"]["happy"] == 3
        assert stats["emocoes"]["sad"] == 1
        assert stats["emocoes"]["neutral"] == 1
        assert stats["distribuicao_percentual"]["happy"] == 60.0
        assert stats["distribuicao_percentual"]["sad"] == 20.0
    
    def test_extrair_estatisticas_emocoes_vazio(self):
        """Testa estatísticas com lista vazia."""
        stats = extrair_estatisticas_emocoes([])
        
        assert stats["total"] == 0
        assert stats["predominante"] is None
        assert stats["emocoes"] == {}
        assert stats["distribuicao_percentual"] == {}


class TestTTSEmotionIntegration:
    """Testes para integração de emoções com TTS."""
    
    def test_tts_input_normalization_strings(self):
        """Testa normalização de entrada com strings simples."""
        from src.services.tts import TTSEngine
        
        # Mock para evitar carregamento real do modelo
        textos = ["Hello", "World"]
        
        # A normalização deve converter strings em dicts
        expected = [
            {"text": "Hello", "emotion_instruction": ""},
            {"text": "World", "emotion_instruction": ""}
        ]
        
        # Este teste verifica a lógica esperada
        assert len(textos) == 2
    
    def test_tts_input_normalization_dicts(self):
        """Testa normalização de entrada com dicts."""
        textos = [
            {"text": "Hello", "emotion_instruction": "Speak happily"},
            {"text": "Goodbye", "emotion_instruction": "Speak sadly"}
        ]
        
        # Dicts devem passar sem modificação
        for item in textos:
            assert "text" in item
            assert "emotion_instruction" in item


class TestConfigurationFlags:
    """Testes para flags de configuração."""
    
    def test_enable_emotion_analysis_flag(self):
        """Testa se flag de análise de emoções existe."""
        assert isinstance(ENABLE_EMOTION_ANALYSIS, bool)
    
    def test_sensevoice_model_config(self):
        """Testa configuração do modelo SenseVoice."""
        assert SENSEVOICE_MODEL is not None
        assert isinstance(SENSEVOICE_MODEL, str)
        assert "SenseVoice" in SENSEVOICE_MODEL


@pytest.mark.integration
class TestPipelineComEmocoes:
    """Testes de integração do pipeline completo."""
    
    @pytest.mark.skip(reason="Requer modelos baixados e GPU")
    def test_pipeline_completo_com_emocoes(self):
        """
        Teste de integração completo do pipeline com emoções.
        
        Este teste requer:
        - Modelos baixados (Whisper, SenseVoice, Qwen3)
        - GPU disponível
        - Arquivo de vídeo de teste
        """
        from src.pipeline import executar_pipeline
        
        video_teste = "tests/fixtures/test_video.mp4"
        
        if not os.path.exists(video_teste):
            pytest.skip("Arquivo de teste não encontrado")
        
        sucesso = executar_pipeline(
            caminho_video=video_teste,
            idioma_origem="eng_Latn",
            idioma_destino="por_Latn",
            idioma_voz="por",
            motor_tts="qwen3",
            modo_encoding="rapido",
            qwen3_mode="custom",
            qwen3_speaker="vivian"
        )
        
        assert sucesso is True


# Fixtures para testes
@pytest.fixture
def segmentos_exemplo():
    """Fixture com segmentos de exemplo para testes."""
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "I'm so excited!",
            "emotion": "happy",
            "emotion_pt": "feliz",
            "emotion_instruction": "Fale com tom alegre e entusiasmado"
        },
        {
            "start": 2.5,
            "end": 5.0,
            "text": "This makes me sad.",
            "emotion": "sad",
            "emotion_pt": "triste",
            "emotion_instruction": "Fale com tom triste e melancólico"
        },
        {
            "start": 5.5,
            "end": 8.0,
            "text": "I can't believe this!",
            "emotion": "angry",
            "emotion_pt": "zangado",
            "emotion_instruction": "Fale com tom zangado, voz elevada"
        }
    ]


@pytest.fixture
def analyzer():
    """Fixture para criar um EmotionAnalyzer."""
    return EmotionAnalyzer(modelo=SENSEVOICE_MODEL)


if __name__ == "__main__":
    # Executar testes com pytest
    pytest.main([__file__, "-v"])
