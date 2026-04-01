import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from resnet import AccidentDetectionModel

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ
# ==========================================
txt_path = Path('Crash-1500.txt')
data_root = Path('Video_Tensors')

crash_labels_dict = {}
if txt_path.exists():
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('vidname'): continue
            parts = line.split(',')
            vid_name = parts[0]
            start_idx = line.find('[')
            end_idx = line.find(']') + 1
            bin_labels = ast.literal_eval(line[start_idx:end_idx])
            crash_labels_dict[vid_name] = bin_labels
else:
    print(f"ПОМИЛКА: Файл анотацій не знайдено за шляхом {txt_path}")

# Збираємо шляхи до .npy файлів та їх мітки
all_paths = []
all_labels = []

# Positive (1500 аварій)
pos_files = sorted(list((data_root / 'positive').glob('*.npy')))
for p in pos_files:
    vid_id = p.stem
    if vid_id in crash_labels_dict:
        all_paths.append(p)
        all_labels.append(crash_labels_dict[vid_id])

# Negative (Звичайний трафік - теж беремо 1500 для балансу)
neg_files = sorted(list((data_root / 'negative').glob('*.npy')))[:1500]
for p in neg_files:
    all_paths.append(p)
    all_labels.append([0] * 50) # 50 нулів

# Розбиття на Train (70%), Val (15%), Test (15%)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42
)

class RawVideoDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        video = np.load(self.file_paths[idx]) # (50, 3, 224, 224)
        
        current_frames = video.shape[0]
        target_frames = 50
        
        if current_frames < target_frames:
            # Якщо кадрів мало, дублюємо останній кадр до 50
            padding_size = target_frames - current_frames
            last_frame = video[-1:]
            padding = np.repeat(last_frame, padding_size, axis=0)
            video = np.concatenate([video, padding], axis=0)
        # elif current_frames > target_frames:
        #     # Якщо раптом більше 50 — обрізаємо
        #     video = video[:target_frames]

        video = torch.from_numpy(video).float() / 255.0
        for t in range(video.size(0)):
            video[t] = self.normalize(video[t])
        

        deltas = torch.zeros_like(video)
        deltas[1:] = video[1:] - video[:-1]       

        combined_video = torch.cat([video, deltas], dim=1)

        # Мітка має бути (50, 1) для BCELoss 
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)
        return combined_video, label

test_loader = DataLoader(RawVideoDataset(test_paths, test_labels), batch_size=8, shuffle=False, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
model = AccidentDetectionModel(hidden_size=256).to(device)
model.load_state_dict(torch.load('models/res_5_best_end_to_end_model_train90.pth', map_location=device))

def test_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Перетворюємо результати в імовірності для метрик
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(loader)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, recall, precision, f1

avg_loss, rec, prec, f1 = test_metrics(model, test_loader, criterion, device)
with open('testdata_results/test_final_tr90.txt', 'a') as f:
    f.write(f"\n--- Avg Test Loss: {avg_loss:.4f} ---\n")
    f.write(f"Recall: {rec:.4f} | Precision: {prec:.4f} \n")
    f.write(f"F1-score: {f1:.4f} \n")
