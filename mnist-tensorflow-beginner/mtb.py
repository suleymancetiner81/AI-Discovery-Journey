import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# MNIST veri setini yükleyelim (hazır geliyor)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Verileri normalize edelim (0-255 arası değerleri 0-1 arasına küçültüyoruz)
x_train = x_train / 255.0 
x_test = x_test / 255.0

# Basit bir model kuralım
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 28x28'lik resmi düzleştiriyoruz
    keras.layers.Dense(128, activation='relu'),  # Gizli katman
    keras.layers.Dense(10, activation='softmax') # Çıkış katmanı (0-9 arası 10 sınıf)
])

# Modeli derleyelim
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitelim
model.fit(x_train, y_train, epochs=5)

# Test verileri üzerinde değerlendirelim
test_loss, test_acc = model.evaluate(x_test, y_test)

print('\nTest doğruluğu:', test_acc)



# Test verilerinden rastgele bir görüntü seçelim
index = np.random.randint(0, len(x_test))
img = x_test[index]
plt.imshow(img, cmap='gray')
plt.show()

# Model tahmin etsin
img = (np.expand_dims(img, 0))  # Modele uygun boyuta getiriyoruz
prediction = model.predict(img)
print("Tahmin edilen rakam:", np.argmax(prediction))