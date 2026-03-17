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
    all_paths, all_labels, test_size=0.3, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42
)

# ==========================================
# 2. КЛАС DATASET
# ==========================================
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
        elif current_frames > target_frames:
            # Якщо раптом більше 50 — обрізаємо
            video = video[:target_frames]

        video = torch.from_numpy(video).float() / 255.0
        for t in range(video.size(0)):
            video[t] = self.normalize(video[t])
        

        deltas = torch.zeros_like(video)
        deltas[1:] = video[1:] - video[:-1]       

        combined_video = torch.cat([video, deltas], dim=1)

        # Мітка має бути (50, 1) для BCELoss 
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)
        return combined_video, label

# Створюємо лоадери
train_loader = DataLoader(RawVideoDataset(train_paths, train_labels), batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(RawVideoDataset(val_paths, val_labels), batch_size=64, shuffle=False, num_workers=0)

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
    def __init__(self, feature_dim=512):
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
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        cnn_out = self.feature_extractor(c_in)
        r_in = cnn_out.view(batch_size, timesteps, 256)
        lstm_out, _ = self.lstm(r_in)
        out = self.classifier(lstm_out)
        return out

def validate_and_metrics(model, loader, criterion, device):
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
    
    return avg_loss, recall, precision

# ==========================================
# 4. ЦИКЛ НАВЧАННЯ
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AccidentDetectionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

best_val_loss = float('inf')

print(f"Навчання розпочато на пристрої: {device}")

for epoch in range(20):
    model.train()
    train_loss = 0
    for i, (videos, labels) in enumerate(train_loader):
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 20 == 0:
            print(f"Епоха {epoch+1}, Батч {i}, Loss: {loss.item():.4f}")

    # Валідація в кінці епохи
    avg_loss, rec, prec = validate_and_metrics(model, val_loader, criterion, device)
    scheduler.step(avg_loss)
    print(f"--- Епоха {epoch+1} ЗАВЕРШЕНА. Avg Val Loss: {avg_loss:.4f} ---")
    print(f"Recall: {rec:.4f} | Precision: {prec:.4f}")
    
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), 'res_5_best_end_to_end_model.pth')
        print("  [NEW BEST MODEL SAVED]")