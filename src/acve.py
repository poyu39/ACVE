import os
import shlex

import demucs.separate
from pyannote.audio import Model
from moviepy.editor import VideoFileClip

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
        # 資料路徑
        self.input_path = f'./../input/{anime_name}'
        self.output_path = f'./../output/{anime_name}'
        os.makedirs(self.output_path, exist_ok=True)
        self.raw_audio_path = f'{self.output_path}/raw_audio'
        os.makedirs(self.raw_audio_path, exist_ok=True)
    
    def video2audio(self, video: str) -> None:
        video_name = video.split('.')[0]
        video_path = f'{self.input_path}/{video}'
        audio_path = f'{self.raw_audio_path}/{video_name}_audio.wav'
        if os.path.exists(audio_path):
            print(f'{audio_path} already exists')
            return audio_path
        print(f'Converting {video_path} to audio')
        print(f'Saving audio to {audio_path}')
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        video_clip.close()
        audio_clip.close()
        return audio_path
    
    def separate(self, audio) -> None:
        demucs.separate.main(shlex.split(f'-n mdx_extra --two-stems vocals --segment 8 -o {self.output_path} {audio}'))
    
    def get_speakers_list(self, audio_path: str) -> list:
        return self.model.get_labels({"uri": audio_path})
    
    def extract(self, audio_path: str) -> dict:
        return self.model({"uri": audio_path})