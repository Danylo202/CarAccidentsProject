import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from sklearn.metrics import recall_score, precision_score
from pathlib import Path
import ast
from resnet import AccidentDetectionModel

data_dict = {}
txt_path = Path('Crash-1500.txt')

with open(txt_path, 'r') as f:
    for line in f:
        if not line.strip() or line.startswith('vidname'): continue
        parts = line.split(',')
        
        vid_name = parts[0]
        # Дістаємо умови (індекси згідно з твоїм описом файлу)
        timing = parts[-3].strip()   # Day/Night
        weather = parts[-2].strip()  # Normal/Snowy/Rainy
        
        # Дістаємо бінарні мітки (вони між [ ])
        start_idx = line.find('[')
        end_idx = line.find(']') + 1
        labels = ast.literal_eval(line[start_idx:end_idx])
        
        data_dict[vid_name] = {
            'timing': timing,
            'weather': weather,
            'labels': labels
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

def evaluate_by_conditions(model, device, data_root):
    model.eval()
    results = []
    
    pos_dir = Path(data_root) / 'positive'
    video_files = sorted(list(pos_dir.glob('*.npy')))
    
    print(f"Починаю аналіз {len(video_files)} відео...")
    
    with torch.no_grad():
        for vid_p in video_files:
            vid_id = vid_p.stem
            if vid_id not in data_dict: continue
            
            video = np.load(vid_p)
            
            if video.shape[0] < 50:
                padding = np.repeat(video[-1:], 50 - video.shape[0], axis=0)
                video = np.concatenate([video, padding], axis=0)
                video = video[:50]
        
            video_tensor = torch.from_numpy(video).float() / 255.0

            deltas = torch.zeros_like(video_tensor)
            deltas[1:] = video_tensor[1:] - video_tensor[:-1]
            
            combined_video = torch.cat([video_tensor, deltas], dim=1)
            
            for t in range(50):
                combined_video[t, :3] = normalize(combined_video[t, :3])
                
            input_tensor = combined_video.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            gt_list = np.array(data_dict[vid_id]['labels'])
            gt_np = np.array(gt_list) # для sklearn recall/precision
            gt_tensor = torch.tensor(gt_list, dtype=torch.float32).unsqueeze(-1).to(device) # для loss

            # Оцінка
            video_loss = criterion(logits.squeeze(0), gt_tensor).item()
            preds = (probs > 0.5).astype(int)
            
            # Рахуємо метрики для цього конкретного відео
            recall = recall_score(gt_np, preds, zero_division=0)
            precision = precision_score(gt_np, preds, zero_division=0)
            
            # Зберігаємо все в один список
            results.append({
                'vid_id': vid_id,
                'timing': data_dict[vid_id]['timing'],
                'weather': data_dict[vid_id]['weather'],
                'loss': video_loss,
                'recall': recall,
                'precision': precision
            })

    return pd.DataFrame(results)

model = AccidentDetectionModel(hidden_size=256).to(device)

model.load_state_dict(torch.load('classic_models/res_5_best_end_to_end_model_dropout0.pth', map_location=device))
df_results = evaluate_by_conditions(model, device, 'Video_Tensors')

with open('models_analyzed/res_5_best_end_to_end_model_dropout0_analysis.txt', 'a') as f:
    f.write("\n--- Результати за часом доби ---")
    f.write(df_results.groupby('timing')[['loss', 'recall', 'precision']].mean().to_string())

    f.write("\n--- Результати за погодою ---")
    f.write(df_results.groupby('weather')[['loss', 'recall', 'precision']].mean().to_string())

    f.write("\n--- Топ найскладніших комбінацій (Loss) ---")
    pivot = df_results.pivot_table(index='weather', columns='timing', values='loss', aggfunc='mean')
    f.write(pivot.to_string())

    f.write("\n--- Топ найскладніших комбінацій (Recall) ---")
    pivot = df_results.pivot_table(index='weather', columns='timing', values='recall', aggfunc='mean')
    f.write(pivot.to_string())

    f.write("\n--- Топ найскладніших комбінацій (Precision) ---")
    pivot = df_results.pivot_table(index='weather', columns='timing', values='precision', aggfunc='mean')
    f.write(pivot.to_string())


