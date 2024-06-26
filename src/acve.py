import os
import shlex
import soundfile as sf
import shutil

from moviepy.editor import VideoFileClip
import demucs.separate
import numpy as np
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines.utils.hook import ProgressHook

from config import CONFIG
from util import time_to_seconds, seconds_to_time


class ACVE:
    def __init__(self, anime_name) -> None:
        '''
            Anime Character Voice Extractor
        '''
        self.embedding_model = Model.from_pretrained('pyannote/embedding', use_auth_token=CONFIG.HUGGINGFACE_ACCESS_TOKEN)
        self.embedding_inference = Inference(self.embedding_model, window="whole")
        
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

    def split_audio(self, audio_path: str, start: float, end: float, save_path) -> None:
        audio, sr = sf.read(audio_path)
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        sf.write(save_path, audio[start_frame:end_frame], sr)
    
    def split_audio_by_dialogue(self, audio_path: str, dialogues: list) -> str:
        audio_name = f"{audio_path.split('/')[-2]}_{audio_path.split('/')[-1].split('.')[0]}"
        output_path = f'{self.output_path}/dialogue_audio/{audio_name}'
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        else:
            print(f'{output_path} already exists')
            return output_path
        now_progress = 1
        for dialogue in dialogues:
            print(f"Splitting {now_progress}/{len(dialogues)} \t {dialogue['start']} ~ {dialogue['end']} \t {dialogue['text']}")
            start: str = dialogue['start']
            end: str = dialogue['end']
            start_seconds = int(time_to_seconds(start)) # 無條件捨去
            end_seconds = int(time_to_seconds(end) + 0.5) # 無條件進位
            output_file_name = f"{output_path}/{seconds_to_time(start_seconds).replace(':', '_')}__{seconds_to_time(end_seconds).replace(':', '_')}"
            self.split_audio(audio_path, start_seconds, end_seconds, f'{output_file_name}.wav')
            with open(f'{output_file_name}.lab', 'w', encoding='utf-8') as lab_file:
                lab_file.write(dialogue['text'])
            now_progress += 1
        return output_path
    
    def speaker_diarization(self, audio_path: str, distance_limit: float, test) -> None:
        audio_name = f"{audio_path.split('/')[-2]}_{audio_path.split('/')[-1].split('.')[0]}"
        embeddings = []
        for audio in os.listdir(audio_path):
            if audio.split('.')[1] == 'wav':
                print(f'Embedding {audio}')
                embeddings.append((audio, self.embedding_inference(f'{audio_path}/{audio}')))
        distances = []
        for i, (audio, embedding) in enumerate(embeddings):
            distances.append([])
            for target_audio, target_embedding in embeddings:
                if audio == target_audio:
                    continue
                distance = cdist(np.reshape(embedding, (1, -1)), np.reshape(target_embedding, (1, -1)), metric='cosine')[0, 0]
                distances[i].append((target_audio, distance))
        speakers = []
        is_classified = [False] * len(embeddings)
        for i, (audio, embedding) in enumerate(embeddings):
            if is_classified[i]:
                continue
            speakers.append([])
            speakers[-1].append(audio)
            for j, distance in enumerate(distances[i]):
                if is_classified[j]:
                    continue
                if distance[1] < distance_limit:
                    # print(f'{audio} and {distance[0]} are the same speaker')
                    speakers[-1].append(distance[0])
                    is_classified[j] = True
        total_audio = 0
        for i, audio in enumerate(speakers):
            total_audio += len(audio)
            print(f'Speaker {i} has {len(audio)} audio')
        print(f'Total {total_audio} audio')
        if test:
            return
        for i, speaker in enumerate(speakers):
            speaker_audio_path = f'{self.output_path}/speaker_audio/{audio_name}/{i}'
            if not os.path.exists(speaker_audio_path):
                os.makedirs(speaker_audio_path, exist_ok=True)
            else:
                print(f'{speaker_audio_path} already exists')
                continue
            for audio in speaker:
                shutil.copy(f'{audio_path}/{audio}', f'{speaker_audio_path}/{audio}')
        print(f'Speaker diarization of {audio_name} is done')