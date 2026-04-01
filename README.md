# 🚗 Traffic Accident Detection and Ego-Involvement Analysis

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
Цей репозиторій показує два підходи до створення моделі комп'ютерного зору для аналізу відео з відеореєстраторів на наявність аварій, а також потенційну участь у ній авто, яке зняло це відео (Ego Involement). 
## Датасет
Моделі натреновані на CarCrashDataset: https://github.com/Cogito2012/CarCrashDataset
Було використано всі 1500 відео з аваріями та 1500 із 3000 нормальних відео (для збереження розподілу).
### 📂 Структура датасету
```markdown
├── Video_Tensors/
│   ├── positive/   # Відео з аваріями
│   └── negative/   # Відео зі звичайним трафіком
├── Crash-1500.txt  # Анотації (назва відео, бінарні мітки кадрів, початковий кадр, youtubeID, час доби, погодні умови, egoinvolve)
```
## Моделі та архітектура
Спочатку були зроблені дві моделі: власна реалізація ResNet та LSTM (resnet.py) та Transfer Learning з основою ResNet18 (resnet2.py) — для однієї задачі: Accident Detection. Дві наступні моделі реалізують 
Multi-Task Learning на основі власної ResNet та ResNet18 для двох задач: оцінки наявності аварії (Accident Detection) та визначення перспективи (Ego Involement): чи бере автомобіль з реєстратором безпосередню 
участь в аварії, чи просто фіксує чуже зіткнення.
### Власна ResNet (resnet.py)
----------
### Transfer Learning з ResNet18 (resnet2.py)
ResNet18 приймає три канали на вхід, на відміну від власної реалізації, тому в лоадерах довелося повертати video, а не combined_video. У класі PretrainedAccidentModel міститься логіка моделі. Спочатку завантажуємо 
ваги ResNet18, потім відрізаємо останній шар (Fully Connected) і залишаємо тільки conv-шари для екстракції фіч. Далі заморожуємо їх. На наступному кроці 512 фіч від ResNet18 йдуть до LSTM (та сама, що й у власній реалізації,
за винятком кількості фіч). У функкції forward відео, розбите на 50 кадрів, пропускається через ResNet18, LSTM та фінальний шар-класифікатор. Використана Loss-функція — BCEWithLogitsLoss, спочатку без, а потім зі
збільшеним штрафом за False negative (pos_weight = [3.0]). Скрипт виводить проміжні результати Loss кожні 20 батчів, а в кінці епохи — ще Precision і Recall. Якщо Loss цієї моделі менший за best_val_loss, то вона буде
збережена у pth-файл.
### Multi-Task Learning на основі власної ResNet (resnet_egoinvolve.py)
Ця модель використовує Multi-Task learning для вирішення двох задач: Accident Detection та Ego-Ivolement Analysis. Порівняно з resnet.py, було змінено процес завантаження даних: тепер з Crash-1500.txt потрібно витягнути
ще й значення egoinvolve. Клас RawVideoDataset повертає три значення: combined_video, label_acc, label_ego. Основа та сама, що й у resnet.py: ResBlock та CustomResidualExtractor залишилися без змін. У класі
AccidentDetectionModel self.classifier замінено на два нові атрибути екземпляра: self.head_accident та self.head_ego. Це дві голови
### Multi-Task Learning на основі ResNet18 (resnet2_egoinvolve.py)
-------
## Оцінка та аналіз моделей
-------
