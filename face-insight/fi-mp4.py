import cv2
import numpy as np

# Yüz tespiti için Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yaş ve cinsiyet tahmini modelleri
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")

# Etiketler
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Video dosyasını aç
cap = cv2.VideoCapture("multiface.mp4")

if not cap.isOpened():
    print("Video açılamadı.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya kare alınamadı.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Çizgi çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Yüz bölgesini al
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.5, 87.5, 114), swapRB=False)

        # Cinsiyet tahmini
        gender_net.setInput(blob)
        gender = gender_list[gender_net.forward().argmax()]

        # Yaş tahmini
        age_net.setInput(blob)
        age = age_list[age_net.forward().argmax()]

        # Yazıyı ekle
        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Ekrana yazdır
    cv2.imshow("Age and Gender Estimation", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Temizle
cap.release()
cv2.destroyAllWindows()