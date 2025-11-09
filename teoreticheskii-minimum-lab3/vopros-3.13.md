# Вопрос №3.13

> Если для каждой точки в датасете заранее известна метка кластера, к которой она принадлежит, то как сопоставить результаты кластеризации с истинными метками?\
> То есть как построить маппинг между метками кластеров и метками классов

Построить маппинг можно оптимально через матрицу совпадений и задачу назначения.

## Шаги (формулы)

1.  Построить матрицу совпадений (contingency):

    $$
    M_{ij}=|{x:\ \hat y(x)=i,\ y(x)=j}|,
    $$

    где $$\hat y$$ — метки кластеров, $$y$$ — истинные метки классов.
2.  Найти перестановку (отображение) $$\pi$$, которая максимизирует число правильно сопоставленных объектов:

    $$
    \max_{\pi}\sum_{i} M_{i,\pi(i)}.
    $$

    Это задача назначения. Её решают алгоритмом Хунгаряна (Hungarian) на матрице затрат. Для минимизации можно взять затраты

    $$
    C = \max(M) - M,
    $$

    и минимизировать $$\sum_i C_{i,\pi(i)}$$.
3. Применить найденное отображение $$\pi$$ к меткам кластеров и посчитать метрики (accuracy, precision/recall/F1, матрицу ошибок и т.д.).

Альтернатива (быстро, нестрого): для каждой кластерной метки присвоить класс с максимальным значением в строке (majority vote). Минус: разные кластеры могут получить одну и ту же целевую метку.

## Вспомогательные оценки (без маппинга)

Используйте метрики, инвариантные к перестановкам меток:

* Adjusted Rand Index (ARI).
* Normalized Mutual Information (NMI) / Adjusted Mutual Information (AMI). Они дают оценку качества без поиска маппинга.

## Purity (простая метрика после соответствия)

$$
\text{purity}=\frac{1}{n}\sum_i \max_j M_{ij}.
$$

## Пример кода (Python, sklearn + scipy)

```python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

y_true = np.array(...)   # истинные метки
y_pred = np.array(...)   # метки кластеров (произвольные значения)

# матрица совпадений: строки = уникальные кластерные метки, столбцы = уникальные true-метки
# sklearn возвращает rows for labels in y_true unique order and cols for y_pred, so we build with swapped args carefully
M = contingency_matrix(y_true, y_pred)  # shape (n_classes, n_clusters)
# Для задачи назначения хотим строки = кластеры, столбцы = классы
M = M.T  # теперь M.shape = (n_clusters, n_classes)

# построить минимизирующую матрицу затрат
cost = M.max() - M
row_ind, col_ind = linear_sum_assignment(cost)

# row_ind[i] (кластер) -> col_ind[i] (класс)
unique_pred = np.unique(y_pred)
unique_true = np.unique(y_true)
mapping = { unique_pred[r]: unique_true[c] for r,c in zip(row_ind, col_ind) }

# применить отображение
y_mapped = np.array([mapping.get(v, -1) for v in y_pred])  # -1 для незамапленных кластеров (если есть)

print("Accuracy (after mapping):", accuracy_score(y_true, y_mapped))
print(classification_report(y_true, y_mapped, zero_division=0))

# purity
purity = M.max(axis=1).sum() / len(y_true)  # если M до transpose, используйте соответствующую ось
print("Purity:", purity)
```

## Практические замечания

* Если $$\#\text{clusters}\ne\#\text{classes}$$, Hungarian корректно работает на прямоугольной матрице (он найдет оптимальную частичную сопоставку). Можно дополнительно дописать фиктивные столбцы/строки.
* Majority-vote даёт допускаемую маппинговую стратегию, но не гарантирует глобальной оптимальности.
* ARI/NMI полезны, когда не хочется или нельзя строить явный маппинг.
* Для стабильности при нескольких запусках кластеризации сравнивать результаты по ARI/NMI и по accuracy после оптимального маппинга.
