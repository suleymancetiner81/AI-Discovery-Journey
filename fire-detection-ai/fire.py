import cv2
import numpy as np

def detect_fire(frame):
    # Renk uzayını BGR'dan HSV'ye çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yangın rengi için HSV aralıkları (kırmızı/turuncu tonları)
    lower_fire = np.array([0, 120, 70])
    upper_fire = np.array([20, 255, 255])
    
    # HSV görüntüde yangın rengini maskele
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Gürültüyü azaltmak için morfolojik işlemler
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def calculate_fire_score(contour, frame_area):
    # Yangın bölgesinin alanını hesapla
    fire_area = cv2.contourArea(contour)
    
    # Yangın alanının tüm frame'e oranını hesapla
    ratio = fire_area / frame_area
    
    # Skor hesapla (0-100 arası)
    score = min(100, ratio * 1000)  # 1000 bir ölçeklendirme faktörü
    
    return score, fire_area

def process_video(video_path):
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    
    # Video özelliklerini al
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    
    # Çıktı videosu için VideoWriter oluştur
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_fire_detection.avi', fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Yangın tespiti yap
        fire_contours = detect_fire(frame)
        
        max_score = 0
        total_fire_area = 0
        
        # Tüm yangın konturlarını işle
        for contour in fire_contours:
            if cv2.contourArea(contour) > 500:  # Küçük bölgeleri göz ardı et
                # Yangın bölgesini dikdörtgenle çerçevele
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # Skor hesapla
                score, fire_area = calculate_fire_score(contour, frame_area)
                total_fire_area += fire_area
                
                # En yüksek skoru tut
                if score > max_score:
                    max_score = score
                
                # Skoru ve alanı ekrana yaz
                cv2.putText(frame, f"Score: {int(score)}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Toplam yangın bilgilerini göster
        if total_fire_area > 0:
            total_ratio = (total_fire_area / frame_area) * 100
            cv2.putText(frame, f"Total Fire: {total_ratio:.2f}%", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Max Danger Score: {int(max_score)}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Sonucu göster
        cv2.imshow('Fire Detection', frame)
        out.write(frame)
        
        # Çıkış için 'q' tuşuna bas
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizlik
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Video dosyasını işle
video_path = 'test.mp4'  # Video dosyanızın yolunu buraya giriniz
process_video(video_path)