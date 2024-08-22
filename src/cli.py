import torch
import argparse
import os

from acve import ACVE
from ass import Ass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime Character Voice Extractor')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing anime files')
    parser.add_argument('--style', type=str, required=True, help='字幕檔中所要擷取的目標風格')
    parser.add_argument('--distance_limit', type=float, required=False)
    parser.add_argument('--test_distance', type=bool, default=False, required=False)
    args = parser.parse_args()
    
    videos = None
    input_path = os.path.abspath(f'./../input/{args.input}')
    
    videos = [os.path.join(file) for file in os.listdir(input_path) if file.split('.')[1] in ['mkv', 'mp4']]
    
    acve = ACVE(args.input)
    for video in videos:
        audio = acve.video2audio(video)
        output_path = acve.separate(audio)
        ass_path = f"{input_path}/{video.split('.')[0]}.ass"
        ass = Ass(ass_path)
        ass.format_events()
        dialogue_audios_path = acve.split_audio_by_dialogue(f'{output_path}/vocals.wav', ass.get_dialogues(args.style))
        # acve.speaker_diarization(dialogue_audios_path, args.distance_limit, args.test_distance)
        # rttm = acve.speaker_diarization(f'{output_path}/vocals.wav')
        # acve.extract_speaker_audio(f'{output_path}/vocals.wav', rttm) 