import os
import shlex
import soundfile as sf

from moviepy.editor import VideoFileClip
import torch
import demucs.separate
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from config import CONFIG


class ACVE:
    def __init__(self, anime_name) -> None:
        '''
            Anime Character Voice Extractor
        '''
        self.pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1',
            use_auth_token=CONFIG.HUGGINGFACE_ACCESS_TOKEN,
        )
        self.pipeline.to(torch.device('cuda'))
        
        # 資料路徑
        self.input_path = f'./../input/{anime_name}'
        self.output_path = f'./../output/{anime_name}'
        os.makedirs(self.output_path, exist_ok=True)
        self.raw_audio_path = f'{self.output_path}/raw_audio'
        os.makedirs(self.raw_audio_path, exist_ok=True)
    
    def video2audio(self, video: str) -> str:
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
    
    def separate(self, audio_path: str) -> str:
        output_path = f"{self.output_path}/mdx_extra/{audio_path.split('/')[-1].split('.')[0]}"
        if os.path.exists(output_path):
            print(f'{audio_path} already separated')
            return output_path
        print(f'Separating {audio_path}')
        demucs.separate.main(
            shlex.split(f'-n mdx_extra --two-stems vocals --segment 8 -o {self.output_path}/ {audio_path}')
        )
        return output_path
    
    def speaker_diarization(self, audio_path: str) -> str:
        audio_name = f"{audio_path.split('/')[-2]}_{audio_path.split('/')[-1].split('.')[0]}"
        rttm_path = f'{self.output_path}/speaker_diarization/{audio_name}.rttm'
        if os.path.exists(rttm_path):
            print(f'{audio_name} already diarized')
            return rttm_path
        print(f'Diarizing {audio_name}')
        with ProgressHook() as hook:
            diarization = self.pipeline(f'{audio_path}', hook=hook)
        
        # 儲存 speaker diarization 結果
        os.makedirs(f'{self.output_path}/speaker_diarization', exist_ok=True)
        with open(rttm_path, 'w') as rttm:
            diarization.write_rttm(rttm)
        return rttm_path

    def split_audio(self, audio_path: str, start: float, duration: float, save_path) -> None:
        audio, sr = sf.read(audio_path)
        start_frame = int(start * sr)
        end_frame = int((start + duration) * sr)
        sf.write(save_path, audio[start_frame:end_frame], sr)
    
    def extract_speaker_audio(self, audio_path: str, rttm_path: str, min_duration = 1) -> None:
        audio_name = f"{audio_path.split('/')[-2]}_{audio_path.split('/')[-1].split('.')[0]}"
        print(f'Extracting speaker audio from {audio_name}')
        with open(rttm_path, 'r') as rttm:
            now_progress = 0
            rttm_lines = rttm.readlines()
            rttm_len = len(rttm_lines)
        for line in rttm_lines:
            _, _, _, start, duration, _, _, speaker, _, _ = line.split()
            if float(duration) < min_duration:
                print(f'skip {speaker}/{start}__{duration} too short')
                continue
            start, duration = float(start), float(duration)
            start_min = f'{int(float(start) // 60)}_{int(float(start) % 60)}' # 轉換成分鐘:秒
            end_min = f'{int((float(start) + float(duration)) // 60)}_{int((float(start) + float(duration)) % 60)}' # 轉換成分鐘:秒
            speaker_path = f'{self.output_path}/speaker_audio/{audio_name}/{speaker}'
            if not os.path.exists(speaker_path):
                os.makedirs(speaker_path, exist_ok=True)
            self.split_audio(audio_path, start, duration,
                            f'{self.output_path}/speaker_audio/{audio_name}/{speaker}/{start_min}__{end_min}.wav')
            print(f'save {speaker}/{start_min}__{end_min}.wav {now_progress}/{rttm_len}')
            now_progress += 1