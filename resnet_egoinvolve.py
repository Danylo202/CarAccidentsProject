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
from sklearn.metrics import recall_score, precision_score

# ==========================================
# 1. ПІДГОТОВКА ДАНИХ
# ==========================================
txt_path = Path('Crash-1500.txt')
data_root = Path('Video_Tensors')

crash_labels_dict = {}
ego_labels_dict = {}
if txt_path.exists():
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('vidname'): continue
            parts = line.split(',')
            vid_name = parts[0]
            start_idx = line.find('[')
            end_idx = line.find(']') + 1
            bin_labels = ast.literal_eval(line[start_idx:end_idx])
            tail = line[end_idx:].strip(',')
            tail_parts = tail.split(',')
            egoinvolve_str = tail_parts[-1].strip()
            ego_label = 1 if egoinvolve_str.lower() == 'yes' else 0
            crash_labels_dict[vid_name] = bin_labels
            ego_labels_dict[vid_name] = [ego_label] * 50
else:
    print(f"ПОМИЛКА: Файл анотацій не знайдено за шляхом {txt_path}")

# Збираємо шляхи до .npy файлів та їх мітки
all_paths = []
all_labels_acc = []
all_labels_ego = []

# Positive (1500 аварій)
pos_files = sorted(list((data_root / 'positive').glob('*.npy')))
for p in pos_files:
    vid_id = p.stem
    if vid_id in crash_labels_dict:
        all_paths.append(p)
        all_labels_acc.append(crash_labels_dict[vid_id])
        all_labels_ego.append(ego_labels_dict[vid_id])

# Negative (Звичайний трафік - теж беремо 1500 для балансу)
neg_files = sorted(list((data_root / 'negative').glob('*.npy')))[:1500]
for p in neg_files:
    all_paths.append(p)
    all_labels_acc.append([0] * 50) # 50 нулів
    all_labels_ego.append([0] * 50) # Якщо немає аварії, то й участь авто неможлива

# Розбиття на Train (80%), Val (10%), Test (10%)
train_paths, temp_paths, train_acc, temp_acc, train_ego, temp_ego = train_test_split(
    all_paths, all_labels_acc, all_labels_ego, test_size=0.2, random_state=42
)

val_paths, test_paths, val_acc, test_acc, val_ego, test_ego = train_test_split(
    temp_paths, temp_acc, temp_ego, test_size=0.5, random_state=42
)

# ==========================================
# 2. КЛАС DATASET
# ==========================================
class RawVideoDataset(Dataset):
    def __init__(self, file_paths, labels_acc, labels_ego):
        self.file_paths = file_paths
        self.labels_acc = labels_acc
        self.labels_ego = labels_ego
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
        label_acc = torch.tensor(self.labels_acc[idx], dtype=torch.float32).unsqueeze(-1)
        label_ego = torch.tensor(self.labels_ego[idx], dtype=torch.float32).unsqueeze(-1)
        return combined_video, label_acc, label_ego

# Створюємо лоадери
train_loader = DataLoader(
    RawVideoDataset(train_paths, train_acc, train_ego), 
    batch_size=8, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    RawVideoDataset(val_paths, val_acc, val_ego), 
    batch_size=8, shuffle=False, num_workers=0
)

# ==========================================
# 3. АРХІТЕКТУРА МОДЕЛІ
# ==========================================

import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Перша згортка
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Друга згортка
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # Якщо вхідний розмір не збігається з вихідним (напр. при stride=2), ми ставимо 1x1 згортку для "підгонки"
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Основний шлях
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # y = x + f(x) 
        out += self.shortcut(x)
        return F.relu(out)

class CustomResidualExtractor(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Початковий шар (Stem) - приймає RGB
        self.stem = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ) # Вихід: (64, 56, 56)

        # Рівень 1: Прості ознаки
        self.layer1 = ResBlock(64, 64, stride=1)
        
        # Рівень 2: Складніші форми (стискаємо вдвічі)
        self.layer2 = ResBlock(64, 128, stride=2) # Вихід: (128, 28, 28)
        
        # Рівень 3: Високорівневі об'єкти (стискаємо ще вдвічі)
        self.layer3 = ResBlock(128, 256, stride=2) # Вихід: (256, 14, 14)

        # Фінальне усереднення (Global Average Pooling)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class AccidentDetectionModel(nn.Module):
    def __init__(self, num_frames=50, feature_dim=256, hidden_size=256):
        super(AccidentDetectionModel, self).__init__()
        
        self.feature_extractor = CustomResidualExtractor(feature_dim)
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, 
                            num_layers=2, batch_first=True, dropout=0.3)

        self.head_accident = nn.Linear(hidden_size, 1) #Чи є аварія взагалі?
        self.head_ego = nn.Linear(hidden_size, 1) # Чи бере участь в аварії?

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        cnn_out = self.feature_extractor(c_in)
        r_in = cnn_out.view(batch_size, timesteps, 256)
        lstm_out, _ = self.lstm(r_in)

        #Вихід для двох задач: аварія та участь в аварії
        out_accident = self.head_accident(lstm_out)
        out_ego = self.head_ego(lstm_out)

        return out_accident, out_ego

def validate_and_metrics(model, loader, criterion_acc, device):
    model.eval()
    total_loss = 0.0
    
    all_preds_acc = []
    all_labels_acc_list = []

    total_correct_ego = 0
    total_acc_frames = 0
    
    with torch.no_grad():
        for videos, labels_acc, labels_ego in loader:
            videos = videos.to(device)
            labels_acc = labels_acc.to(device)
            labels_ego = labels_ego.to(device)
            
            out_acc, out_ego = model(videos)

            # Рахуємо Loss аварії
            loss_acc = criterion_acc(out_acc, labels_acc)
            
            # Рахуємо Loss участі (з маскою)
            loss_ego_all = F.binary_cross_entropy_with_logits(out_ego, labels_ego, reduction='none')
            mask = (labels_acc == 1).float()
            
            if mask.sum() > 0:
                loss_ego = (loss_ego_all * mask).sum() / mask.sum()
                batch_loss = loss_acc + loss_ego
            else:
                batch_loss = loss_acc
                
            total_loss += batch_loss.item()
            
            # Збираємо дані для Recall/Precision (тільки для аварій)
            probs_acc = torch.sigmoid(out_acc)
            preds_acc = (probs_acc > 0.5).cpu().numpy()
            
            all_preds_acc.extend(preds_acc.flatten())
            all_labels_acc_list.extend(labels_acc.cpu().numpy().flatten())
            
            # Рахуємо точність egoinvolve тільки для кадрів з реальною аварією
            bool_mask = (labels_acc == 1) # Булева маска для індексації
            
            if bool_mask.sum() > 0:
                probs_ego = torch.sigmoid(out_ego)
                preds_ego = (probs_ego > 0.5).float()
                
                # Відфільтровуємо тільки ті кадри, де є аварія
                valid_preds_ego = preds_ego[bool_mask]
                valid_labels_ego = labels_ego[bool_mask]
                
                # Рахуємо кількість збігів
                total_correct_ego += (valid_preds_ego == valid_labels_ego).sum().item()
                total_acc_frames += bool_mask.sum().item()
                
    avg_loss = total_loss / len(loader)
    recall = recall_score(all_labels_acc_list, all_preds_acc, zero_division=0)
    precision = precision_score(all_labels_acc_list, all_preds_acc, zero_division=0)
    
    ego_accuracy = (total_correct_ego / total_acc_frames) * 100 if total_acc_frames > 0 else -1.0
    
    return avg_loss, recall, precision, ego_accuracy

if __name__ == '__main__':
    # ==========================================
    # 4. ЦИКЛ НАВЧАННЯ
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AccidentDetectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Критерій тільки для аварії 
    criterion_acc = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

    best_val_loss = float('inf')

    print(f"Навчання MTL-моделі розпочато на пристрої: {device}")

    # for epoch in range(20):
    #     model.train()
    #     train_loss = 0
    #     for i, (videos, labels_acc, labels_ego) in enumerate(train_loader):
    #         videos = videos.to(device)
    #         labels_acc = labels_acc.to(device)
    #         labels_ego = labels_ego.to(device)
            
    #         optimizer.zero_grad()
    #         out_acc, out_ego = model(videos)
    #         # 1. Помилка детекції аварії (рахується завжди)
    #         loss_acc = criterion_acc(out_acc, labels_acc)
            
    #         # 2. Помилка участі в ДТП (рахується тільки для кадрів з реальною аварією)
    #         loss_ego_all = F.binary_cross_entropy_with_logits(out_ego, labels_ego, reduction='none')
    #         mask = (labels_acc == 1).float()           
    #         if mask.sum() > 0:
    #             loss_ego = (loss_ego_all * mask).sum() / mask.sum()
    #             total_loss = loss_acc + loss_ego
    #         else:
    #             total_loss = loss_acc
    #         total_loss.backward()
    #         optimizer.step()

    #         train_loss += total_loss.item()
    #         if i % 40 == 0:
    #             print(f"Епоха {epoch+1}, Батч {i}, Loss: {total_loss.item():.4f}")

    #     # Валідація в кінці епохи
    #     avg_val_loss, rec, prec, ego_acc = validate_and_metrics(model, val_loader, criterion_acc, device)
    #     scheduler.step(avg_val_loss)
        
    #     print(f"--- Епоха {epoch+1} ЗАВЕРШЕНА. Avg Val Loss: {avg_val_loss:.4f} ---")
    #     print(f"Recall (Acc): {rec:.4f} | Precision (Acc): {prec:.4f} | Ego Acc: {ego_acc:.2f}%")
        
    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), 'res_5_mtl_best_model.pth')
    #         print("  [NEW BEST MODEL SAVED]")

    # ==========================================
    # 5. ФІНАЛЬНА ОЦІНКА НАЙКРАЩОЇ МОДЕЛІ
    # ==========================================

    print("\n[INFO] Завершення циклу епох. Завантаження найкращої MTL моделі...")

    final_model = AccidentDetectionModel().to(device)

    model_path = 'res_5_mtl_best_model.pth'
    final_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"[SUCCESS] Ваги з '{model_path}' завантажено успішно.")

    final_model.eval()

    print("[INFO] Розрахунок фінальних метрик на Validation Set...")
    with torch.no_grad():
        final_val_loss, final_recall, final_precision, final_ego_acc = validate_and_metrics(
            final_model, val_loader, criterion_acc, device
        )
    # Збереження результатів у текстовий файл 
    results_file = "best_mtl_model_results.txt"
    with open(results_file, 'a', encoding="utf-8") as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"MODEL: AccidentDetectionModel (MTL: Accident + Ego)\n")
        f.write(f"SETTINGS: LR=5e-5, Batch=8, Pos_Weight=3.0\n")
        f.write(f"--- RESULTS (Accident Detection) ---\n")
        f.write(f"Final Val Loss:  {final_val_loss:.4f}\n")
        f.write(f"Final Recall:    {final_recall:.4f}\n")
        f.write(f"Final Precision: {final_precision:.4f}\n")
        f.write(f"Final Ego Involve Accuracy: {final_ego_acc:.2f}%\n")
        f.write("="*50 + "\n")

    #вивід у консоль
    print("\n" + "★" * 40)
    print("      ФІНАЛЬНІ РЕЗУЛЬТАТИ (MTL MODEL)")
    print(" " + "★" * 40)
    print(f"  Loss:      {final_val_loss:.4f}")
    print(f"  Recall:    {final_recall:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Ego Involve Accuracy: {final_ego_acc:.2f}%")
    print(" " + "★" * 40)
    print(f"[INFO] Результати збережено у файл: {results_file}\n")