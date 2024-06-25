import torch
import argparse
import os

from acve import ACVE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime Character Voice Extractor')
    parser.add_argument('--input', type=str, required=False, help='Path to the directory containing anime files')
    args = parser.parse_args()
    
    videos = None
    input_path = os.path.abspath(f'./../input/{args.input}')
    
    videos = [os.path.join(video) for video in os.listdir(input_path)]
    
    acve = ACVE(args.input)
    for video in videos:
        audio = acve.video2audio(video)
        acve.separate(audio)