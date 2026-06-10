import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = './asl_alphabet_train'
IMG_SIZE = 64

print("1. Загрузка данных с жесткого диска...")

# Генератор для обучения: нормализует (1/255) и делает аугментацию (крутит, сдвигает)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    validation_split=0.2    # Откладывает 20% картинок для теста
)

# Генератор для проверки: ТОЛЬКО нормализует (искажать тестовые данные нельзя!)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("2. Подключаю обучающую выборку...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

print("3. Подключаю тестовую выборку...")
validation_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Сохраняем название папки, как список который понимает модель
classes = list(train_generator.class_indices.keys())
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(classes)

with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)

print("4. Создаю CNN...")
model = Sequential([
    #Глаза модели (Conv2D) сужаются до 3х3 пикселя и просматривают все изображение, в три слоя(32, 64, 128 фильтров)
    #BatchNormalization() - стабилизирует яркость и контраст
    #MaxPooling2D((2, 2)) - выбирает самый яркий признак из квадрата 2х2
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(), #из квадрата делает длинный плоский список
    Dense(128, activation='relu'), #128 нейронов анализирует этот список
    BatchNormalization(),
    Dropout(0.5),#отключает 50% нейронов, для того что бы предотвратить переобучение и сделать оставшиеся нейроны сильнее

    Dense(len(classes), activation='softmax') #буква с наибольшей вероятностью становиться ответом
])


print("5. Начинаю обучение...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Запускаем конвейер по 32 фото, 15 эпох и проверка
model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)

print("6. Сохраняю модель...")
model.save('cnn_model.h5')

print(f"ГОТОВО! Нейросеть обучена")