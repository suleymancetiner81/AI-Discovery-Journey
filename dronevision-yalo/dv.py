from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle (n - nano model: hızlı ve hafif)
model = YOLO("yolov8n.pt")

# Video dosyasını aç
video_path = "video4.mp4"
cap = cv2.VideoCapture(video_path)

# Video çözünürlük bilgisi
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video yazıcı (istersen çıkış videosu kaydetmek için)
# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Model ile tahmin yap
    results = model(frame)



    # Tahmin sonucunu çizdir
    annotated_frame = results[0].plot()

    scale = 0.5  # %50 küçült
    resized_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)

    # Küçük boyutta ekranda göster
    cv2.imshow("Drone Nesne Tespiti - YOLOv8", resized_frame)





    # Kaydetmek istersen:
    # out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Temizlik
cap.release()
# out.release()
cv2.destroyAllWindows()