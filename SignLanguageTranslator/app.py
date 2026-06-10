import customtkinter as ctk
import cv2
from PIL import Image
import pickle
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# настройка визуала
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# создание класса приложения
class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Переводчик Языка Жестов")
        self.geometry("900x600")

        # загрузка модели
        self.model = tf.keras.models.load_model('cnn_model.h5')
        self.label_encoder = pickle.load(open('label_encoder.pickle', 'rb'))

        # настройка зрения модели
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

        # создание визуальных объектов и кнопок
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.btn_webcam = ctk.CTkButton(self.sidebar, text="Включить камеру", command=self.start_webcam, height=40)
        self.btn_webcam.pack(pady=20, padx=10)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="Выключить", fg_color="#c0392b", hover_color="#e74c3c",
                                      command=self.stop_stream, height=40)
        self.btn_stop.pack(pady=10, padx=10)

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.main_frame, text="Камера выключена", width=640, height=480,
                                        fg_color="black", corner_radius=10)
        self.video_label.pack(pady=10)

        self.result_label = ctk.CTkLabel(self.main_frame, text="Покажи жест", font=("Arial", 40, "bold"),
                                         text_color="#2ecc71")
        self.result_label.pack(pady=10)

        self.cap = None

        self.prediction_buffer = deque(maxlen=15)

    def start_webcam(self):
        self.stop_stream()
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def stop_stream(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.configure(image=None, text="Камера выключена")
        self.result_label.configure(text="Покажи жест")
        cv2.destroyAllWindows()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # чтение кадра, смена цветового кода в RGB
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Находим границы руки (рамку)
                        x_min, y_min = w, h
                        x_max, y_max = 0, 0

                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min, y_min = min(x_min, x), min(y_min, y)
                            x_max, y_max = max(x_max, x), max(y_max, y)

                        # Делаем отступы, чтобы рука влезла целиком
                        pad = 30
                        x_min = max(0, x_min - pad)
                        y_min = max(0, y_min - pad)
                        x_max = min(w, x_max + pad)
                        y_max = min(h, y_max + pad)

                        # Рисуем зеленый квадрат вокруг руки
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Вырезаем руку из кадра
                        hand_crop = frame[y_min:y_max, x_min:x_max]

                        if hand_crop.size != 0:
                            # сжатие с сохранением пропорций
                            h_crop, w_crop, _ = hand_crop.shape
                            max_side = max(h_crop, w_crop)

                            # Создаем идеальный квадрат по размеру длинной стороны
                            square_img = np.zeros((max_side, max_side, 3), dtype=np.uint8)

                            # Вычисляем отступы, чтобы вклеить руку по центру
                            y_offset = (max_side - h_crop) // 2
                            x_offset = (max_side - w_crop) // 2

                            square_img[y_offset:y_offset + h_crop, x_offset:x_offset + w_crop] = hand_crop

                            # Окошко отладки показывает идеальный квадрат
                            cv2.imshow("Eyes of CNN", square_img)
                            cv2.waitKey(1)

                            # Сжимаем идеальный квадрат, пальцы не искажаются
                            resized = cv2.resize(square_img, (64, 64))
                            rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            normalized = rgb_resized / 255.0
                            reshaped = np.reshape(normalized, (1, 64, 64, 3))

                            # Нейросеть угадывает
                            prediction = self.model.predict(reshaped, verbose=0)

                            # фильтр уверенности
                            max_confidence = np.max(prediction)
                            predicted_idx = np.argmax(prediction)

                            # Обновляем результат только если модель уверена больше чем на 80%
                            if max_confidence > 0.80:
                                predicted_char = self.label_encoder.inverse_transform([predicted_idx])[0]

                                # Стабилизатор: добавляем в память и берем самую частую
                                self.prediction_buffer.append(predicted_char)
                                most_common_char = Counter(self.prediction_buffer).most_common(1)[0][0]

                                self.result_label.configure(
                                    text=f"Буква: {most_common_char} ({int(max_confidence * 100)}%)")
                            else:
                                # Если нейросеть сомневается, просим показать лучше
                                self.result_label.configure(text="Жду четкий жест...")

                # обновляем картинку в интерфейсе, или отключаем ее отображение
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))

                self.video_label.configure(image=img_tk, text="")
                self.video_label.image = img_tk

                self.after(1, self.update_frame)
            else:
                self.stop_stream()


if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()