# 🚗 Traffic Accident Detection and Ego-Involvement Analysis

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
Цей репозиторій показує два підходи до створення моделі комп'ютерного зору для аналізу відео з відеореєстраторів на наявність аварій, а також потенційну участь у ній авто, яке зняло це відео (Ego Involement). 
## Датасет
Моделі натреновані на CarCrashDataset: https://github.com/Cogito2012/CarCrashDataset
Було використано всі 1500 відео з аваріями та 1500 із 3000 нормальних відео (для збереження розподілу).
## Моделі та архітектура
Спочатку були зроблені дві моделі: власна реалізація ResNet та LSTM (resnet.py) та Transfer Learning з основою ResNet18 (resnet2.py) — для однієї задачі: Accident Detection. Дві наступні моделі реалізують 
Multi-Task Learning на основі власної ResNet та ResNet18 для двох задач: оцінки наявності аварії (Accident Detection) та визначення перспективи (Ego Involement): чи бере автомобіль з реєстратором безпосередню 
участь в аварії, чи просто фіксує чуже зіткнення.
### Власна ResNet (resnet.py)
----------
### Transfer Learning з ResNet18 (resnet2.py)
------
### Multi-Task Learning на основі власної ResNet (resnet_egoinvolve.py)
-------
### Multi-Task Learning на основі ResNet18 (resnet2_egoinvolve.py)
-------
## Оцінка та аналіз моделей
-------
