""" % The whole code was written in Jupyter Notebook and exported into .py extension %  

    % This code is written to classify the type of brain tumours (Glioma, Meningioma, Pituitary and Notumour) by taking the input of MRI scanned images. There are two models trained in the code where one model uses no data augmentation while other model uses data augmentation %  

    % Please see at the end for the references used % """

# % All the necessary libraries imported % #
import cv2
import numpy as np
import PIL
import pathlib
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers

# % Specification of image dataset’s directory location %
training_data_dir = ("archive/Training")
training_data_dir = pathlib.Path(training_data_dir)
testing_data_dir = ("archive/Testing")
testing_data_dir = pathlib.Path(testing_data_dir)

# % Confirmation of the image count for train & test %
training_image_count = len(list(training_data_dir.glob('*/*.jpg')))
print('Number of training images are:', training_image_count)
testing_image_count = len(list(testing_data_dir.glob('*/*.jpg')))
print('Number of testing images are:', testing_image_count)

# % Plotting the Train-Test image count %
plt.figure(figsize=(5,5))
plt.bar(('Train', 'Test'), (training_image_count, testing_image_count), width=0.2)
plt.title('Train-Test image count')
plt.ylabel('Image count')
plt.show()

# % Retrieving images of specific categories %
tumour_types = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumour']
notumor_images = list(training_data_dir.glob('notumor/*'))
glioma_images = list(training_data_dir.glob('glioma/*'))
meningioma_images = list(training_data_dir.glob('meningioma/*'))
pituitary_images = list(training_data_dir.glob('pituitary/*'))

# % Visualization of dataset according to the type of classes %
plt.figure(figsize=(5,5))
colours = ['red', 'green', 'blue', 'black']
plt.bar(tumour_types, (len(glioma_images), len(meningioma_images), len(pituitary_images), len(notumor_images)), width=0.5, color=colours)
plt.title('Number of Classes with image count')
plt.xlabel('Tumour Type', fontsize=14)
plt.ylabel('Image Count', fontsize=14)
plt.show()

# % Sample images from all classes %
PIL.Image.open(str(meningioma_images[1]))
PIL.Image.open(str(glioma_images[1]))
PIL.Image.open(str(pituitary_images[1]))
PIL.Image.open(str(notumor_images[1]))

# % Images from each tumour types stored in train/test dictionaries %
training_images_dict = {
    'glioma': list(training_data_dir.glob('glioma/*')),
    'meningioma': list(training_data_dir.glob('meningioma/*')),
    'pituitary': list(training_data_dir.glob('pituitary/*')),
    'notumor': list(training_data_dir.glob('notumor/*'))
}
testing_images_dict = {
    'glioma': list(testing_data_dir.glob('glioma/*')),
    'meningioma': list(testing_data_dir.glob('meningioma/*')),
    'pituitary': list(testing_data_dir.glob('pituitary/*')),
    'notumor': list(testing_data_dir.glob('notumor/*'))
}

# % Class labels %
labels_dict = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2,
    'notumor': 3
}

# % Pre-processing of train dataset %
X_train, y_train = [], []

for tumor_type, images in training_images_dict.items():
    for image in images:
        # Retrieval of image pixels in array format using OpenCV library
        img = cv2.imread(str(image))

        # Resize of image pixels into equal sizes
        resized_img = cv2.resize(img, (180, 180))

        # Encoding of classes into similar indexes
        X_train.append(resized_img)
        y_train.append(labels_dict[tumor_type])

# % Pre-processing of test dataset %
X_test, y_test = [], []

for tumor_type, images in testing_images_dict.items():
    for image in images:
        # Retrieval of image pixels in array format using OpenCV library
        img = cv2.imread(str(image))

        # Resize of image pixels into equal sizes
        resized_img = cv2.resize(img, (180, 180))

        # Encoding of classes into similar indexes
        X_test.append(resized_img)
        y_test.append(labels_dict[tumor_type])

# % Conversion of pixel arrays into NumPy array %
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# % Array normalization %
X_train = X_train / 255
X_test = X_test / 255

# % Model without data augmentation is implemented at first % #
print('\n Model without data augmentation is implemented at first')

# % Creation of CNN model using 3 convolutional layers, Dropout & 2 dense layers %
noAug_model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

# % Model is complied using Adam optimizer with loss function %
noAug_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# % Training of model %
model_hist = noAug_model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1, batch_size=16, epochs=10)

# % Model is evaluated and checked for validation accuracy %
print('\n Model Evaluation')
noAug_model.evaluate(X_test, y_test)


# % Saving the model %
noAug_model.save('withoutAug_model.h5')

# % Prediction of model %
predictions = noAug_model.predict(X_test)
prediction_labels = [np.argmax(i) for i in predictions]

# % Confusion matrix of predictions %
cm = tf.math.confusion_matrix(labels=y_test,predictions=prediction_labels)
# Plotting of CM
plt.figure(figsize = (10,7))
plt.title('Confusion matrix without Data Augmentation')
sn.heatmap(cm, annot=True, fmt='d')
plt.xticks(np.arange(4), training_images_dict.keys())
plt.yticks(np.arange(4), training_images_dict.keys())
plt.xlabel('Predicted Values')
plt.ylabel('True Values')

# % A report of classification is generated %
print('\n Classification Report without Data Augmentation')
print(classification_report(y_test, prediction_labels, target_names=training_images_dict.keys()))

# % A plot of accuracy vs epochs %
hist_list = model_hist.history
list_ep = [i for i in range(1,11)]
plt.figure(figsize=(10,10))
plt.title('Accuracy vs Epochs without Data Augmentation')
plt.plot(list_ep, hist_list['accuracy'], ls = '--', label='accuracy')
plt.plot(list_ep, hist_list['val_accuracy'],ls = '--', label='val_accuracy')
plt.ylabel('Accuracy', fontsize = 15)
plt.xlabel('Epochs', fontsize = 15)
plt.legend()
plt.show()

# % A plot of loss vs epochs %
plt.figure(figsize=(10,10))
plt.title('Loss vs Epochs without Data Augmentation')
plt.plot(list_ep, hist_list['loss'], ls = '--', label='loss')
plt.plot(list_ep, hist_list['val_loss'],ls = '--', label='val_loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Model is then trained with Data Augmentation to reduce the overfitting and improve the accuracy.
print('\n Model is then trained with Data Augmentation to reduce the overfitting and improve the accuracy')

# % Data Augmentation layer initiated %
data_aug = keras.Sequential([    
    layers.experimental.preprocessing.RandomFlip('horizontal'), 
    layers.experimental.preprocessing.RandomContrast(0.9)
])

# % Creation of CNN model using augmented layer, 3 convolutional layers, Dropout & 2 dense layers %
aug_model = Sequential([
    data_aug,
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

# % Model is complied using Adam optimizer with loss function %
aug_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# % Training of model %
model_hist = aug_model.fit(X_train, y_train, validation_data=(X_test,y_test), verbose=1, batch_size=32, epochs=20)

# % Model is evaluated and checked for validation accuracy %
print('\n Model Evaluation')
aug_model.evaluate(X_test, y_test)

# % Summary of the model parameters %
print('\n Model Summary')
aug_model.summary()

# % Saving the model %
aug_model.save('withAug_model.h5')

# % Prediction of model %
predictions = aug_model.predict(X_test)
prediction_labels = [np.argmax(i) for i in predictions]

# % Confusion matrix of predictions %
cm = tf.math.confusion_matrix(labels=y_test,predictions=prediction_labels)
# Plotting of CM
plt.figure(figsize=(10,7))
plt.title('Confusion matrix with Data Augmentation')
sn.heatmap(cm, annot=True, fmt='d')
plt.xticks(np.arange(4), training_images_dict.keys())
plt.yticks(np.arange(4), training_images_dict.keys())
plt.xlabel('Predicted Values', fontsize=15)
plt.ylabel('True Values', fontsize=15)

# % A report of classification is generated %
print('\n Classification Report with Data Augmentation')
print(classification_report(y_test, prediction_labels, target_names=training_images_dict.keys()))

# % A plot of accuracy vs epochs %
hist_list = model_hist.history
list_ep = [i for i in range(1,21)]
plt.figure(figsize=(10,10))
plt.title('Accuracy vs Epochs with Data Augmentation')
plt.plot(list_ep, hist_list['accuracy'], ls = '--', label='accuracy')
plt.plot(list_ep, hist_list['val_accuracy'],ls = '--', label='val_accuracy')
plt.ylabel('Accuracy', fontsize = 15)
plt.xlabel('Epochs', fontsize = 15)
plt.legend()
plt.show()

# % A plot of loss vs epochs %
plt.figure(figsize=(10,10))
plt.title('Loss vs Epochs with Data Augmentation')
plt.plot(list_ep, hist_list['loss'], ls = '--', label='loss')
plt.plot(list_ep, hist_list['val_loss'],ls = '--', label='val_loss')
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.legend()
plt.show()


# % References % #

# 1. Codebasics Youtube channel. Data augmentation to address overfitting | Deep Learning Tutorial 26 (Tensorflow, Keras & Python).

# 2. CSC-40070 Applications of AI, Machine Learning and Data Science. Lab 11: Image recognition using deep learning networks.

# 3. M. Nickparvar, “Brain Tumor MRI Dataset,” Kaggle, 2021. [Online]. Available: 10.34740/kaggle/dsv/2645886. [Accessed 22 March 2022].

# 4. KNOWLEDGE DOCTOR Youtube channel. Brain Tumor Detection Using Deep Learning | Python Tensorflow Keras.