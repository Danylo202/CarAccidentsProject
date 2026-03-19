import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
import torchvision.models as models
import torch.nn as nn

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

        # Мітка має бути (50, 1) для BCELoss 
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)
        return video, label

class PretrainedAccidentModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(PretrainedAccidentModel, self).__init__()
        
        # Завантажуємо переднавчену ResNet18 
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Відрізаємо останній шар (fc), залишаємо тільки conv-шари для екстракції фіч
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Заморожуємо всі ваги ResNet, щоб не зламати їх на старті
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # LSTM: на вхід іде 512 фіч від ResNet
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                            num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        cnn_out = self.feature_extractor(c_in) #Вихід: (batch*time, 512, 1, 1)
        cnn_out = cnn_out.view(cnn_out.size(0), -1) #Вихід: (batch*time, 512)
        r_in = cnn_out.view(batch_size, timesteps, -1)
        
        lstm_out, _ = self.lstm(r_in)
        out = self.classifier(lstm_out)
        return out
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
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

    # Створюємо лоадери
    train_loader = DataLoader(RawVideoDataset(train_paths, train_labels), batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(RawVideoDataset(val_paths, val_labels), batch_size=16, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedAccidentModel().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss() 

    import os
    results_file = "best_model_results.txt"
    best_val_loss = float('inf')

    #Перед початком навчання намагаємося прочитати попередній найкращий Val Loss з файлу результатів
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if "Validation Loss:" in line:
                    loss_str = line.split(":")[1].strip()
                    best_val_loss = float(loss_str)
                    print(f"[INFO] Знайдено попередній найкращий Val Loss: {best_val_loss}")
                    break
    else:
        print("[INFO] Файл результатів не знайдено. Починаємо з inf.")

    print(f"Навчання Pretrained моделі розпочато на пристрої: {device}")
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
        
        #Валідація
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for v_videos, v_labels in val_loader:
                v_videos, v_labels = v_videos.to(device), v_labels.to(device)
                v_outputs = model(v_videos)
                val_loss += criterion(v_outputs, v_labels).item()
                v_outputs = torch.sigmoid(v_outputs)
                preds = (v_outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(v_labels.cpu().numpy().flatten())
                
        avg_val_loss = val_loss / len(val_loader)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
    
        print(f"--- Епоха {epoch+1} ЗАВЕРШЕНА. Avg Val Loss: {avg_val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} ---")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_pretrained_model.pth')
            print("  [NEW BEST MODEL SAVED]")
            
            # Збереження результатів найкращої моделі у txt файл
            with open("best_pretrained_model_results.txt", "w", encoding="utf-8") as f:
                f.write("=== Найкраща модель ===\n")
                f.write(f"Епоха: {epoch+1}\n")
                f.write(f"Validation Loss: {best_val_loss:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")