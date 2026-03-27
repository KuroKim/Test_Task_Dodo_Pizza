import cv2
import argparse
import pandas as pd
from ultralytics import YOLO


def is_overlapping(box1, box2):
    """
    Проверяет пересечение двух прямоугольников (Bounding Boxes).
    Формат:[x1, y1, x2, y2]
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return False
    return True


def main(video_path):
    print("Загрузка модели YOLOv8...")
    # Используем yolov8n (nano) - она самая быстрая и легкая
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    # Получаем свойства видео для сохранения результата
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Настраиваем сохранение видео (output.mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        return

    print("Выделите столик мышкой и нажмите ПРОБЕЛ или ENTER.")
    roi = cv2.selectROI("Select Table ROI", frame, showCrosshair=False)
    cv2.destroyWindow("Select Table ROI")

    rx, ry, rw, rh = roi
    table_box = [rx, ry, rx + rw, ry + rh]
    print(f"Выбрана зона стола: {table_box}")

    # --- ПЕРЕМЕННЫЕ СОСТОЯНИЯ (STATE MACHINE) ---
    current_state = "Unknown"  # "Empty" или "Occupied"
    events = []  # Журнал событий для Pandas

    # Буферы ("терпение"), чтобы избежать ложных срабатываний, если нейросеть моргнула на 1 кадр
    PATIENCE_FRAMES = int(fps * 1.5)  # Ждем 1.5 секунды для подтверждения статуса
    empty_counter = 0
    occupied_counter = 0

    print("Начинаем обработку видео... Нажмите 'q' для досрочной остановки.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Текущее время в секундах
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 1. ДЕТЕКЦИЯ: Ищем людей (classes=[0] - это person в YOLO)
        results = model(frame, classes=[0], verbose=False)

        person_in_zone = False

        # 2. АНАЛИЗ: Проверяем каждого найденного человека
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Получаем координаты людей[x1, y1, x2, y2]
            for box in boxes:
                # Рисуем рамку человека (синяя)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

                # Проверяем, пересекается ли человек со столом
                if is_overlapping(table_box, box):
                    person_in_zone = True

        # 3. ЛОГИКА: Обновляем статус стола с учетом буфера (защита от "морганий")
        if person_in_zone:
            empty_counter = 0
            occupied_counter += 1
            if occupied_counter > PATIENCE_FRAMES and current_state != "Occupied":
                current_state = "Occupied"
                events.append({"time_sec": current_time_sec, "event": "Occupied (Approach)"})
                print(f"[{current_time_sec:.1f}s] Событие: К столу подошли (Занят)")
        else:
            occupied_counter = 0
            empty_counter += 1
            if empty_counter > PATIENCE_FRAMES and current_state != "Empty":
                current_state = "Empty"
                events.append({"time_sec": current_time_sec, "event": "Empty (Left)"})
                print(f"[{current_time_sec:.1f}s] Событие: Из-за стола ушли (Свободен)")

        # 4. ВИЗУАЛИЗАЦИЯ: Рисуем стол и его статус
        color = (0, 0, 255) if current_state == "Occupied" else (0, 255, 0)  # Красный = Занят, Зеленый = Свободен
        if current_state == "Unknown": color = (0, 255, 255)  # Желтый на старте

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 3)
        cv2.putText(frame, f"State: {current_state}", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Пишем кадр в итоговое видео
        out.write(frame)

        # Показываем на экране с уменьшенным размером (чтобы влезло в монитор)
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Table Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрываем все потоки
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nОбработка завершена. Файл output.mp4 сохранен.")

    # --- 5. АНАЛИТИКА PANDAS ---
    if not events:
        print("Событий не зафиксировано.")
        return

    df = pd.DataFrame(events)
    print("\n--- Журнал событий ---")
    print(df)

    # Считаем среднее время между 'Empty' и следующим 'Occupied'
    delays = []
    empty_time = None

    for index, row in df.iterrows():
        if row['event'] == 'Empty (Left)':
            empty_time = row['time_sec']
        elif row['event'] == 'Occupied (Approach)' and empty_time is not None:
            delay = row['time_sec'] - empty_time
            delays.append(delay)
            empty_time = None  # Сброс до следующего ухода

    print("\n--- Результаты Аналитики ---")
    if delays:
        avg_delay = sum(delays) / len(delays)
        print(f"Зафиксировано {len(delays)} циклов уборки/посадки.")
        print(f"Среднее время простоя столика: {avg_delay:.2f} секунд.")
    else:
        print("Полных циклов (Уход -> Подход) не зафиксировано.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table Cleaning Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    main(args.video)
