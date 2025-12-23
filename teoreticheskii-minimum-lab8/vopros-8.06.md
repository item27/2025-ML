# Вопрос №8.06

> [Transposed convolution](https://programmersought.com/article/1224509238/). Чем отличается от обычной свертки. Как позволяет увеличивать feature map. Гиперпараметры слоя [convTransposed2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)

### Transposed convolution (ConvTranspose2d)

**Transposed convolution** — это операция, которую удобно понимать как “обратную по форме” к обычной свёртке: она часто используется, когда нужно **увеличить** пространственное разрешение feature map (например, $$14\times14 \rightarrow 28\times28$$). В PyTorch она описана как градиент операции $$Conv2d$$ по входу (то есть тесно связана с тем, как считается backprop для свёрток), и это **не** настоящая математическая “обратная свёртка”. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

***

### Чем отличается от обычной свёртки

#### Обычная свёртка (Conv2d)

* Берёт локальные окна входной карты признаков и считает взвешенную сумму ядром.
* При $$stride>1$$ обычно **уменьшает** размер карты признаков (downsampling).

#### Transposed convolution (ConvTranspose2d)

* Делает так, чтобы из меньшей карты признаков получить большую (upsampling).
* Ключевая механика при $$stride>1$$: **вставляет нули между элементами входа по пространственным осям**, а затем применяет свёрточное ядро. Это и даёт увеличение разрешения, причём “апсемплинг” становится **обучаемым** (ядро обучается). ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

Интуиция: обычная свёртка “сжимает” и смешивает соседние пиксели, а transposed conv “размазывает” каждый входной элемент по большему выходному полю через ядро и перекрытия.

***

### Как она увеличивает feature map

Если $$stride=2$$:

1. между соседними элементами входной feature map вставляются “пустые” позиции (нули);
2. дальше выполняется обычная свёртка по этой “разреженной” карте;
3. за счёт разреживания сетки и перекрытия ядра выход получается больше по $$H,W$$. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

Важно: увеличение размера зависит не только от $$stride$$, но и от $$kernel_size$$, $$padding$$, $$dilation$$ и $$output_padding$$ (ниже — формула).

***

### Формула размера выхода в PyTorch

Для входа размера $$(N, C_{in}, H_{in}, W_{in})$$ выход имеет $$(N, C_{out}, H_{out}, W_{out})$$, где: ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

$$H_{out}=(H_{in}-1)\times stride[0]-2\times padding[0]+dilation[0]\times (kernel_size[0]-1)+output_padding[0]+1$$

$$W_{out}=(W_{in}-1)\times stride[1]-2\times padding[1]+dilation[1]\times (kernel_size[1]-1)+output_padding[1]+1$$

***

### Гиперпараметры слоя `torch.nn.ConvTranspose2d`

Сигнатура: $$ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')$$ ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

Что означает каждый параметр:

* **`in_channels`** — число входных каналов $$C_{in}$$. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`out_channels`** — число выходных каналов $$C_{out}$$. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`kernel_size`** — размер ядра $$k_h\times k_w$$. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`stride`** — шаг; при $$stride>1$$ происходит вставка нулей между входными элементами, что даёт апсемплинг. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`padding`** — “необычный” параметр в transposed conv: в документации он описан как добавление неявного нулевого паддинга в количестве $$dilation\times (kernel_size-1)-padding$$ по сторонам (это сделано, чтобы при одинаковых параметрах $$Conv2d$$ и $$ConvTranspose2d$$ согласовывались по формам). ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`output_padding`** — добавляет размер **к одной стороне** выходной формы (нужно для разрешения неоднозначности форм при $$stride>1$$); при этом это **не** “доп. нули в выходе”, а именно настройка вычисляемого размера. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`dilation`** — расстояние между элементами ядра (à trous). ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`groups`** — группировка каналов (как в обычных свёртках); $$in_channels$$ и $$out_channels$$ должны делиться на $$groups$$. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`bias`** — добавлять ли обучаемый сдвиг. ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))
* **`padding_mode`** — режим паддинга (для ConvTranspose2d в документации указан `'zeros'`). ([docs.pytorch.org](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html))

***

### Практический нюанс: “checkerboard artifacts”

Transposed convolution может давать “шахматные” артефакты из-за неравномерных перекрытий при апсемплинге; часто альтернативой используют “resize (nearest/bilinear) + Conv2d”. ([distill.pub](https://distill.pub/2016/deconv-checkerboard))
