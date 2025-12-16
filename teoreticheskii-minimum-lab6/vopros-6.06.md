# Вопрос №6.06

> Как использовать RNN-ячейки для обработки спектрограмм? Например, как реализовать классификацию аудио?

### Как использовать RNN-ячейки для обработки спектрограмм

#### 1) Общая идея

Спектрограмма (или mel-спектрограмма) — это матрица признаков **во времени**:

* $$X \in \mathbb{R}^{T \times F}$$ где $$T$$ — число временных кадров (окон), $$F$$ — число частотных бинов или mel-каналов.

RNN (LSTM/GRU) умеет обрабатывать **последовательности**, поэтому мы трактуем спектрограмму так:

* **каждый момент времени** $$t$$ — это вектор $$x_t \in \mathbb{R}^{F}$$
* последовательность $$x_0, x_1, \dots, x_{T-1}$$ подаём в RNN.

То есть RNN “читает” аудио кадр за кадром и накапливает контекст.

***

### 2) Типовой пайплайн классификации аудио с RNN

#### Шаг A — извлечь признаки

Варианты:

* log-mel спектрограмма $$T \times F$$ (часто $$F=64/80/128$$)
* MFCC $$T \times 13$$ (или $$39$$ с дельтами)

Часто применяют нормализацию:

* по всему датасету (mean/std),
* или по каждому примеру (CMVN).

#### Шаг B — подать в RNN

* Input: $$[B, T, F]$$ (batch-first)
* RNN: GRU/LSTM → выдаёт скрытые состояния $$H \in \mathbb{R}^{B \times T \times H}$$

#### Шаг C — агрегировать по времени

Чтобы получить один вектор на весь аудиофайл, есть несколько стандартных способов:

1. **Последнее состояние** (если однонаправленная RNN): $$h = H[:, T-1, :]$$
2. **Пулинг по времени**: $$h = \mathrm{mean}_t(H)$$ или $$h = \mathrm{max}_t(H)$$
3. **Attention-пулинг** (лучше, но сложнее): модель сама “выбирает” важные кадры.

#### Шаг D — классификатор

Полносвязный слой:

* logits: $$z = W h + b$$
* вероятности: $$p=\mathrm{softmax}(z)$$
* loss: кросс-энтропия.

***

### 3) Важные нюансы (чтобы реально работало)

#### Переменная длина аудио

Аудио разной длины ⇒ $$T$$ разный. Решения:

* паддинг + маска и `pack_padded_sequence` (PyTorch),
* или нарезка/обрезка до фиксированной длины,
* или пулинг/attention с mask.

#### RNN не любит очень длинные $$T$$

Если hop=10мс, то 10 секунд = 1000 кадров. Часто делают:

* downsampling (увеличить hop),
* или перед RNN ставят 2D/1D Conv, чтобы уменьшить $$T$$ и извлечь локальные паттерны.

***

### 4) Пример реализации в PyTorch (mel → GRU → mean-pooling → classifier)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioRNNClassifier(nn.Module):
    def __init__(self, n_mels: int, hidden: int, n_classes: int, n_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_mels,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,   # часто лучше
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # x: [B, T, F]
        if lengths is not None:
            # pack для переменной длины
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.rnn(x)  # out: [B, T, 2H]

        # mean pooling по времени (с учётом lengths, если есть)
        if lengths is None:
            h = out.mean(dim=1)  # [B, 2H]
        else:
            B, T, D = out.shape
            mask = torch.arange(T, device=out.device).unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
            out = out * mask.unsqueeze(-1)
            h = out.sum(dim=1) / lengths.unsqueeze(1)  # [B, 2H]

        logits = self.fc(h)  # [B, C]
        return logits
```

Обучение:

* вход: mel-спектрограммы $$[B,T,F]$$
* loss: `F.cross_entropy(logits, y)`
* оптимизатор: Adam.

***

### 5) Архитектурные варианты

#### Вариант 1: чистая RNN (baseline)

* mel/MFCC → GRU/LSTM → pooling → FC Просто и быстро, но может быть слабее на шумных данных.

#### Вариант 2: CNN + RNN (часто лучше)

* Conv2D по spectrogram (извлекает локальные паттерны: гармоники, форманты)
* затем RNN по времени
* pooling/attention → FC

#### Вариант 3: BiRNN + Attention

* BiLSTM/GRU
* attention pooling вместо mean/last Лучше выделяет “полезные” моменты (например, плач среди тишины).

***

### 6) Мини-чеклист для классификации аудио

* признаки: log-mel $$F=64/80$$
* окно: $$25$$мс, hop: $$10$$мс
* нормализация (mean/std)
* BiGRU/BiLSTM (hidden 128–256)
* pooling: mean или attention
* аугментации: шум, time shift, specaugment (по желанию)
