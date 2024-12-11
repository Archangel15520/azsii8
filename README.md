# Практика 8: Методы защиты от атак на модели ИИ

**Цель задания:**

Изучить методы защиты моделей ИИ от различных атак, включая методы защиты на уровне данных,
моделирования и обучения. Реализовать эти методы и проверить их эффективность против атак,
изученных ранее.

**Задачи:**

1. Изучить и реализовать защиту модели с помощью тренировок на противоречивых примерах
(Adversarial Training).
2. Реализовать метод защиты на основе градиентной маскировки.
3. Использовать регуляризацию и нормализацию для повышения устойчивости модели.
4. Проверить эффективность методов защиты против атак FGSM, PGD и GAN-based атак.
5. Оценить улучшение точности защищенной модели на противоречивых примерах.
Шаги выполнения:

  ---

**WARNING(Важная информация): 1. Все работы по данному предмету можно найти по ссылке: https://github.com/Archangel15520/AZSII-REPO/tree/main**

**2. В коде используется ранее обученная модель на датасете MNIST, которую можно найти в закрепе к данному проекту.**

**3. Сылка на выполненую работу в среде google colab: https://colab.research.google.com/drive/1rpVqhjkuz0KI56Egtfsjgh77lPopM0bV?usp=sharing** 

  ---
  
## Шаг 1: Защита с помощью Adversarial Training

Adversarial Training — это метод защиты, который заключается в том, чтобы обучать модель на
противоречивых примерах. Этот метод помогает модели научиться быть более устойчивой к атакам,
так как она сталкивается с противоречивыми примерами на этапе обучения.

```
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images[:1000] / 255.0
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels[:1000], 10)
    return train_images, train_labels, test_images, test_labels

def create_fgsm_adversary(image, epsilon, gradient):
    perturbed_image = image + epsilon * np.sign(gradient)
    return np.clip(perturbed_image, 0, 1)

def generate_adversarial_batch(model, images, labels, epsilon):
    adversarial_images = []
    for image, label in zip(images, labels):
        image_tensor = tf.convert_to_tensor(image.reshape(1, 28, 28, 1), dtype=tf.float32)
        label_tensor = tf.convert_to_tensor(label.reshape(1, 10), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = model(image_tensor)
            loss = tf.keras.losses.categorical_crossentropy(label_tensor, predictions)
        gradients = tape.gradient(loss, image_tensor)
        adversarial_image = create_fgsm_adversary(image_tensor.numpy(), epsilon, gradients.numpy()).reshape(28, 28)
        adversarial_images.append(adversarial_image)
    return np.array(adversarial_images)

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_with_adversarial_data(model, train_images, train_labels, epsilon):
    for epoch in range(5):
        for start_idx in range(0, len(train_images), 32):
            batch_images = train_images[start_idx:start_idx + 32]
            batch_labels = train_labels[start_idx:start_idx + 32]
            adversarial_images = generate_adversarial_batch(model, batch_images, batch_labels, epsilon)
            augmented_images = np.concatenate((batch_images, adversarial_images))
            augmented_labels = np.concatenate((batch_labels, batch_labels))
            model.train_on_batch(augmented_images, augmented_labels)

def main():
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data()
    model = build_model()
    train_with_adversarial_data(model, train_images, train_labels, epsilon=0.1)
    model.save('adversarial_trained_model.h5')

if __name__ == "__main__":
    main()
```

![image](https://github.com/Archangel15520/azsii8/blob/main/screenshot/1.JPG)

## Шаг 2: Градиентная маскировка (Gradient Masking)

Gradient Masking — это метод защиты, который затрудняет доступ к градиентам модели для атак. Он
используется для уменьшения информации, доступной для атакующих, и усложнения поиска
направленных изменений.

```
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    return train_images, train_labels, test_images, test_labels

def build_masked_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softplus')  # Использование softplus вместо softmax
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=5)
    return model

def main():
    train_images, train_labels, _, _ = preprocess_data()
    masked_model = build_masked_model()
    trained_model = train_model(masked_model, train_images, train_labels)
    trained_model.save('masked_model.h5')

if __name__ == "__main__":
    main()
```

![image](https://github.com/Archangel15520/azsii8/blob/main/screenshot/2.JPG)

## Шаг 3: Регуляризация и нормализация для повышения устойчивости

Использование таких методов, как L2-регуляризация, дропаут и нормализация батчей, может помочь
улучшить устойчивость модели к атакам.

```
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    return train_images, train_labels, test_images, test_labels

def build_regularized_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(model, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=5)
    model.save('regularized_model.h5')

def main():
    train_images, train_labels, _, _ = preprocess_data()
    regularized_model = build_regularized_model()
    train_and_save_model(regularized_model, train_images, train_labels)

if __name__ == "__main__":
    main()
```

![image](https://github.com/Archangel15520/azsii8/blob/main/screenshot/3.JPG)

## Шаг 4: Оценка моделей на противоречивых примерах

Теперь проверим эффективность всех защитных методов на атакованных данных, созданных с
помощью FGSM и других методов, таких как PGD или GAN.

```
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def evaluate_on_adversarial_data(models, adversarial_images, true_labels):
    for model_name, model in models.items():
        loss, accuracy = model.evaluate(adversarial_images, true_labels, verbose=0)
        print(f"Точность модели \"{model_name}\" на противоречивых примерах: {accuracy:.4f}")

def main():
    # Загрузка предварительно обученных моделей
    model_paths = {
        'защищенная (protected)': 'adversarial_trained_model.h5',
        'регуляризованная (regularized)': 'regularized_model.h5',
        'с маскировкой (masked)': 'masked_model.h5'
    }
    models = {name: load_model(path) for name, path in model_paths.items()}

    # Генерация противоречивых примеров (FGSM)
    _, _, test_images, test_labels = preprocess_data()
    adversarial_images = generate_adversarial_batch(models['защищенная (protected)'], test_images[:100], test_labels[:100], epsilon=0.1)

    # Оценка моделей на противоречивых данных
    evaluate_on_adversarial_data(models, adversarial_images, test_labels[:100])

if __name__ == "__main__":
    main()
```

![image](https://github.com/Archangel15520/azsii8/blob/main/screenshot/4.JPG)

**Вывод:**

В ходе эксперимента протестированы три подхода к защите моделей от атак с использованием противоречивых примеров: обучение на противоречивых данных, регуляризация с нормализацией и градиентная маскировка. Для оценки устойчивости использовались атакованные примеры, созданные методом FGSM.

Результаты показали, что модель с градиентной маскировкой продемонстрировала наибольшую точность — 71.00%, немного опередив модель с регуляризацией и нормализацией, которая достигла 68.00%. Защищённая модель, обученная на противоречивых примерах, показала наименьшую точность — 55.00%.

Таким образом, наиболее устойчивой к атакам оказалась модель с градиентной маскировкой, однако методы регуляризации и нормализации также показали высокий потенциал для защиты от атак.
