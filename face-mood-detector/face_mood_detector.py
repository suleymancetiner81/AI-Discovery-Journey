import cv2
from deepface import DeepFace

emoji_map = {
    'happy': '😄', 'sad': '😢', 'angry': '😠',
    'surprise': '😲', 'neutral': '😐', 'fear': '😱',
    'disgust': '🤢'
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emoji = emoji_map.get(emotion, '')
        cv2.putText(frame, f"{emotion} {emoji}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    except Exception as e:
        print("Yüz algılanamadı:", e)

    cv2.imshow("Face Mood Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()