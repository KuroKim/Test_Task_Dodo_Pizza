import cv2
import argparse
import pandas as pd
from ultralytics import YOLO


def main(video_path):
    # 1. Загружаем готовую легковесную модель YOLOv8n (n = nano, самая быстрая)
    print("Загрузка модели YOLOv8...")
    model = YOLO('yolov8n.pt')  # Модель скачается автоматически при первом запуске

    # 2. Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    # 3. Читаем первый кадр для выбора зоны (ROI - Region of Interest)
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        return

    # Пользователь выделяет столик мышкой
    print("Выделите столик мышкой и нажмите ПРОБЕЛ или ENTER.")
    roi = cv2.selectROI("Select Table ROI", frame, showCrosshair=False)
    cv2.destroyWindow("Select Table ROI")

    # roi это кортеж: (x, y, width, height)
    rx, ry, rw, rh = roi
    print(f"Выбрана зона стола: x={rx}, y={ry}, w={rw}, h={rh}")

    # Здесь мы будем настраивать сохранение нового видео (VideoWriter)
    # Здесь будем хранить историю событий для Pandas

    # Главный цикл обработки кадров
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Видео закончилось

        # --- ТУТ БУДЕТ МАГИЯ НЕЙРОСЕТИ ---
        # 1. model(frame, classes=[0]) -> находим только людей
        # 2. Проверяем пересечение координат людей с (rx, ry, rw, rh)
        # 3. Меняем цвет рамки стола (Красный/Зеленый)
        # 4. Пишем время смены статуса в лог

        # Показываем кадр на экране (для отладки)
        cv2.imshow("Table Detection", frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- ТУТ БУДЕТ PANDAS АНАЛИТИКА ---
    # Считаем среднее время и выводим в консоль


if __name__ == "__main__":
    # Настраиваем чтение аргументов из консоли
    parser = argparse.ArgumentParser(description="Table Cleaning Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    main(args.video)
