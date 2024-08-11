import os
import argparse

def add_prefix(folder_path, prefix):
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            new_file_name = prefix + file
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)
            print(f'已將檔案 {file} 重新命名為 {new_file_name}')

def padding_prefix(folder_path):
    files = os.listdir(folder_path)
    for file in files:
        episode = str(file).split('_')[0]
        if len(episode) == 1:
            # 1 to 01
            old_file_path = os.path.join(folder_path, file)
            file = f'0{episode}_{file}'
            new_file_path = os.path.join(folder_path, file)
            os.rename(old_file_path, new_file_path)
            print(f'已將檔案 {old_file_path} 重新命名為 {new_file_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='欲更名檔案所在的資料夾')
    # parser.add_argument('--prefix', help='新檔案名稱的前綴')
    args = parser.parse_args()
    # add_prefix(args.folder, args.prefix)
    padding_prefix(args.folder)