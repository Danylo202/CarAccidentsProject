import cv2
import os
from pathlib import Path

input_base = Path('Videos')
output_base = Path('Photos')

# Категорії, які ми хочемо обробити
categories = ['positive', 'negative']

for category in categories:
    video_folder = input_base / category
    photo_folder = output_base / category
    
    # Перевіряємо, чи існує папка з відео
    if not video_folder.exists():
        print(f"Пропускаю {category}: папка не знайдена.")
        continue
        
    # Створюємо відповідну папку в Photos
    photo_folder.mkdir(parents=True, exist_ok=True)
    
    # Знаходимо всі mp4 файли
    video_files = list(video_folder.glob('*.mp4'))
    print(f"\n--- Обробка категорії: {category} ({len(video_files)} відео) ---")
    
    for video_path in video_files:
        video_name = video_path.stem
        # Створюємо підпапку для кожного конкретного відео
        video_output_dir = photo_folder / video_name
        video_output_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        saved_count = 0
        
        # Відображаємо прогрес у терміналі
        print(f"  Процесинг {video_name}...", end='\r')
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Вихід, якщо відео закінчилось або вже є 50 кадрів
            if not ret or saved_count >= 50:
                break
                
            file_path = video_output_dir / f"{saved_count:02d}.jpg"
            cv2.imwrite(str(file_path), frame)
            saved_count += 1
                
            frame_count += 1
            
        cap.release()
