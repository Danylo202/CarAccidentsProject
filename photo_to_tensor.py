import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Шлях, де лежать нарізані кадри
input_base = Path(r'Photos') 
# Куди збережемо готові тензори
output_base = Path(r'Video_Tensors')

categories = ['positive', 'negative']

# Базовий ресайз під стандарт CNN (224x224, без нормалізації)
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor() # Залишає uint8, але міняє на (3, H, W)
])

def process_videos():
    for cat in categories:
        cat_path = input_base / cat
        out_cat_path = output_base / cat
        out_cat_path.mkdir(parents=True, exist_ok=True)
        
        # Відсортований список папок-відео (000001, 000002...)
        video_folders = sorted([f for f in cat_path.iterdir() if f.is_dir()])
        
        print(f"Обробка категорії: {cat} ({len(video_folders)} відео)")
        
        for vid_folder in video_folders:
            frames = []
            # Беремо всі 50 кадрів
            img_files = sorted(list(vid_folder.glob('*.jpg')))
            
            for img_p in img_files:
                img = Image.open(img_p).convert('RGB')
                tensor_img = resize_transform(img) # Форма (3, 224, 224)
                frames.append(tensor_img.numpy())
            
            # Склеюємо в один масив відео: (50, 3, 224, 224)
            video_array = np.stack(frames)
            
            # Зберігаємо як компактний .npy файл
            np.save(out_cat_path / f"{vid_folder.name}.npy", video_array)
            
            if int(vid_folder.name) % 100 == 0:
                print(f"  Оброблено {vid_folder.name}...")

process_videos()