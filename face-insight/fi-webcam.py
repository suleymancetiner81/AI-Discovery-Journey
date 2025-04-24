import cv2
import numpy as np

# OpenCV'nin hazır yüz tespit modeli (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cinsiyet ve yaş tahmini için modelleri yükle
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")

# Yaş ve cinsiyet için sınıf etiketleri
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Kamerayı başlat (0 = varsayılan webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    # Gri tonlamaya çevir (yüz tespiti için daha uygun)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Tespit edilen yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Yüzün bölgesini al
        face = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.5, 87.5, 114), swapRB=False)

        # Yaş tahminini yap
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds.argmax()]

        # Cinsiyet tahminini yap
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds.argmax()]

        # Sonuçları ekranda göster
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Ekrana görüntüyü bastır
    cv2.imshow("Face Detection - Age and Gender Prediction", frame)

    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
