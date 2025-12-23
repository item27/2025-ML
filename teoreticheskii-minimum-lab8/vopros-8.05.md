# Вопрос №8.05

> Модель диффузии на [упрощенном примере](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion) линейной диффузии. Из каких моделей состоит, как обучается

### Линейная диффузия (упрощённый пример)

Идея диффузионных моделей: мы учимся **убирать шум** так, чтобы из шума постепенно получались данные (картинки). В классическом виде есть два процесса:

* **Прямой (forward)**: берём реальную картинку и постепенно добавляем шум, получая всё более шумную версию.
* **Обратный (reverse)**: учим модель по шумной версии предсказывать, какой шум добавили (или как выглядит “чистая” картинка), чтобы шаг за шагом “очищать” шум и в итоге получить картинку. ([Sander Dieleman](https://sander.ai/2023/07/20/perspectives.html?utm_source=chatgpt.com))

В статье про _Linear Diffusion_ это показывают на MNIST и максимально упрощают компоненты: вместо нейросетей — линейные модели. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))

***

### Из каких моделей/блоков состоит (в линейном примере)

1. **Text embedding (условие по тексту)** Текстовый промпт — это просто цифра `"0"..."9"`. Его кодируют **one-hot** вектором (условие). ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
2. **Image encoder/decoder (переход в латентное пространство и обратно)** Картинку $$28 \times 28$$ переводят в более короткий вектор (“латент”) через **PCA**, а обратно восстанавливают обратным преобразованием PCA. В реальных диффузиях часто роль энкодера/декодера играет VAE, но здесь PCA как линейный аналог. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
3. **Noising (добавление шума)** В латент добавляют гауссов шум: $$\text{latent_noisy} = \text{latent} + \epsilon,;;\epsilon \sim \mathcal{N}(0, I)$$ В статье это делается буквально сложением с шумом из стандартного нормального распределения. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
4. **Denoiser (модель, которая предсказывает шум)** В “настоящих” диффузиях это обычно U-Net, предсказывающий шум. Здесь — **линейная регрессия**. На вход ей подают:

* шумный латент,
* текстовый one-hot,
* и **interaction terms** (произведения “текст × латент”), чтобы линейной модели было проще связать условие с тем, какой шум характерен для этой цифры. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))

5. **Шаг восстановления (subtract noise)** Предсказали шум $$\hat{\epsilon}$$ и вычли: $$\text{latent_denoised} = \text{latent_noisy} - \hat{\epsilon}$$ Потом декодируем латент обратно в изображение через обратный PCA. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))

***

### Как обучается (в линейном примере)

Обучение — это просто supervised-задача “предсказать шум”:

1. Берём реальную картинку $$x$$ и её метку (цифру).
2. Кодируем картинку в латент: $$z = \text{PCA_encode}(x)$$.
3. Генерируем шум $$\epsilon \sim \mathcal{N}(0, I)$$ и делаем $$z_{noisy} = z + \epsilon$$. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
4. Собираем признаки для денойзера: $$\text{features} = [z_{noisy},; \text{text_onehot},; \text{interactions}]$$, где interactions — поэлементные произведения “каждый компонент one-hot × латент”. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
5. Цель (таргет) — тот самый шум $$\epsilon$$. Обучаем линейную регрессию приближать: $$\hat{\epsilon} = f(\text{features})$$ ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))

***

### Как происходит генерация

Ключевой трюк: **вместо латента реальной картинки стартуем с чистого шума в латентном пространстве**.

1. Берём промпт `"5"` → one-hot.
2. Берём $$z_{start} = \epsilon,;\epsilon \sim \mathcal{N}(0, I)$$ (это “шум как латент”). ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
3. Прогоняем через денойзер и получаем $$\hat{\epsilon}$$.
4. Делаем “очищенный латент”: $$z_{denoised} = z_{start} - \hat{\epsilon}$$ ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
5. Декодируем через обратный PCA → получаем картинку цифры.

***

### Что упрощено относительно “настоящей” диффузии

* В реальных моделях шум добавляют **по шагам** (много $$t$$), и денойзинг тоже идёт много шагов с расписанием шума (noise schedule). В статье отмечено, что настоящие диффузии обычно “вычитают шум по чуть-чуть и повторяют много раз”, а линейная версия вычитает сразу. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
* Вместо U-Net — линейная регрессия, поэтому качество ограничено, но структура процесса сохраняется. ([Count Bayesie](https://www.countbayesie.com/blog/2023/4/21/linear-diffusion))
