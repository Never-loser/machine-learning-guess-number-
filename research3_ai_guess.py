import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def preprocess_image(image_path, img_size=(28, 28)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = cv2.bitwise_not(img)
    img = img / 255.0
    return img.reshape(img_size[0], img_size[1], 1)
def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def train_with_mnist():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0
    model = build_model()
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
    model.save('digit_recognition.h5')
    return model
def predict_digit(model, image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(np.array([img]))
        return str(np.argmax(prediction))
    except Exception as e:
        return f"خطا: {str(e)}"
def recognize_digit_with_pretrained(image_path):
    try:
        model = tf.keras.models.load_model('digit_recognition.h5')
        return predict_digit(model, image_path)
    except:
        import easyocr
        reader = easyocr.Reader(['en'])
        result = reader.readtext(image_path, allowlist='0123456789', detail=0)
        return result[0] if result else "عددی تشخیص داده نشد"
image_path = 'your_image.png'
print(f"عدد تشخیص داده شده: {recognize_digit_with_pretrained(image_path)}")
