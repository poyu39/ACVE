import os
import shutil
import argparse

def filter_lab_files(source, target):
    if not os.path.exists(source):
        raise FileNotFoundError(f"來源資料夾 '{source}' 不存在。")
    if not os.path.exists(target):
        os.makedirs(target)

    for root, _, files in os.walk(source):
        for file in files:
            if file.endswith('.lab'):
                file_path = os.path.join(root, file)
                shutil.move(file_path, target)
                print(f"已將 {file_path} 移動至 {target}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='從來源資料夾篩選出 .lab 檔案並移動至目標資料夾。')
    parser.add_argument('--source', type=str, help='包含 .lab 檔案的來源資料夾路徑。')
    parser.add_argument('--target', type=str, help='將 .lab 檔案移動至的目標資料夾路徑。')
    args = parser.parse_args()

    filter_lab_files(args.source, args.target)
