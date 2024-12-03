import os
import urllib.request

def download_yolo_files():
    # Create models/ml directory if it doesn't exist
    os.makedirs('models/ml', exist_ok=True)
    
    files = {
        'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    for filename, url in files.items():
        filepath = os.path.join('models/ml', filename)
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            urllib.request.urlretrieve(url, filepath)
            print(f'Downloaded {filename}')
        else:
            print(f'{filename} already exists')

if __name__ == "__main__":
    download_yolo_files()