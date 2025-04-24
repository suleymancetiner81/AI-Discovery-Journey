# Python ile Yapay Zeka Destekli Gerçek Zamanlı (Kamera - Stream) Yaş ve Cinsiyet Tahmini Uygulaması

Bu yazımda, Python ve OpenCV kullanarak bir webcam görüntüsü üzerinden gerçek zamanlı yüz tespiti ile yaş ve cinsiyet tahmini yapan bir yapay zeka uygulaması geliştirdim. Hadi şimdi kodları adım adım ele alalım ve bu tarz uygulamaların nerelerde kullanılacağını ve nasıl geliştirileceğini birlikte değerlendirelim…

## 🚀 Giriş: Bu Proje Nedir?

Bu proje, gerçek zamanlı olarak yüzleri tespit edip, her yüz için yaş ve cinsiyet tahmini yapabilen bir uygulamadır. Bilgisayarınızın kamerası üzerinden gelen görüntüyü analiz eder ve tahmin sonuçlarını ekran üzerinde anlık olarak gösterir.

Bu uygulama neler yapabilir?

✔️ Yüzleri algılar
✔️ Her yüz için yaş aralığını tahmin eder
✔️ Cinsiyeti tahmin eder
✔️ Sonuçları görsel olarak ekran üzerine yansıtır


## Kullanılan Teknolojiler

Bu projede kullanılan kütüphaneler:

✔️ **Python:** Çalıştırılacak platform
✔️ **OpenCV:** Görüntü işleme ve kamera erişimi için
✔️ **Deep Learning modelleri:** Önceden eğitilmiş caffe formatında modeller



## 🚀 Gerekli Dosyalar ve Kurulum

Aşağıdaki dosyaları aynı klasöre indirmeniz gerekiyor:

✔️ haarcascade_frontalface_default.xml
✔️ deploy_age.prototxt, age_net.caffemodel
✔️ deploy_gender.prototxt, gender_net.caffemodel


## 🚀 Kurulması gereken Python kütüphaneleri:

✔️ opencv-python
✔️ numpy


## Yazar

[**Süleyman Çetiner**](https://medium.com/@suleymancetiner81) tarafından yazılmıştır.

---

Bu projede kullanılan teknikler ve yöntemler hakkında daha fazla bilgi almak için [Medium yazısını](https://suleymancetiner81.medium.com/python-ile-yapay-zeka-destekli-ger%C3%A7ek-zamanl%C4%B1-kamera-stream-ya%C5%9F-ve-cinsiyet-tahmini-uygulamas%C4%B1-5908964be75c) ziyaret edebilirsiniz.
