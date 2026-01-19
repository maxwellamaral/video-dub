
import pytest
import os
import shutil
from moviepy import ColorClip, AudioArrayClip
import numpy as np

@pytest.fixture(scope="session")
def input_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("input")
    return d

@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("output")
    return d

@pytest.fixture(scope="session")
def synthetic_video(input_dir):
    """Cria um vídeo dummy de 5 segundos com áudio nulo/silêncio."""
    filename = input_dir / "test_video.mp4"
    str_path = str(filename)
    
    # Criar vídeo simples (cor sólida)
    duration = 5
    fps = 24
    
    # Clip de vídeo (Azul)
    video = ColorClip(size=(640, 360), color=(0, 0, 255), duration=duration)

    # Gerar áudio (senoidal simples para ter algo)
    from moviepy import AudioClip
    def make_frame_audio(t):
        return [np.sin(440 * 2 * np.pi * t), np.sin(440 * 2 * np.pi * t)]
    
    audio = AudioClip(make_frame_audio, duration=duration, fps=44100)
    video = video.with_audio(audio)
    
    video.write_videofile(str_path, fps=fps, codec="libx264", audio_codec="aac", logger=None)
    
    return str_path

@pytest.fixture(autouse=True)
def mock_directories(monkeypatch, output_dir):
    """Redireciona OUTPUT_DIR para pasta temporária."""
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    # Precisamos monkeypatchar a variavel global no modulo pipeline_dublagem se ela for usada diretamente
    # Importar aqui para evitar execucao prematura
    import sys
    sys.path.append(os.getcwd())
