import os

import demucs.separate
import shlex
from pyannote.audio import Model
from audio_extract import extract_audio

from config import CONFIG


class ACVE:
    def __init__(self, anime_name) -> None:
        '''
            Anime Character Voice Extractor
        '''
        self.model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=CONFIG.HUGGINGFACE_ACCESS_TOKEN,
        )
        self.output_path, self.raw_audio_path = self.create_output_directory(f'./../output/{anime_name}')
    
    def create_output_directory(self, output_path: str) -> tuple:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/raw_audio', exist_ok=True)
        return output_path, f'{output_path}/raw_audio'
    
    def video2audio(self, video_path: str) -> None:
        print(f'Extracting audio from {video_path}')
        audio_path = f"{self.raw_audio_path}/{video_path.split('/')[-1].split('.')[0]}_audio.wav"
        extract_audio(input_path=video_path, output_path=audio_path)
    
    def separate(self, audio_path: str, output_path: str) -> None:
        demucs.separate.separate(
            shlex.split(f"--dl -n demucs -d cpu {audio_path} {output_path}")
        )
    
    def get_speakers_list(self, audio_path: str) -> list:
        return self.model.get_labels({"uri": audio_path})
    
    def extract(self, audio_path: str) -> dict:
        return self.model({"uri": audio_path})