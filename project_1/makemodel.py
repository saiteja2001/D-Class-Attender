# from copyreg import pickle
# import os,numpy,cv2,joblib, re
# import pickletools
# from pickle import PickleError
#
# haar_file = 'haarcascade_frontalface_default.xml'
# datasets = 'media'
# print('Training...')
# (images, labels, names, id) = ([], [], {}, 0)
#
#
# for (subdirs, dirs, files) in os.walk(datasets):
#     for subdir in dirs:
#         names[id] = subdir
#         subjectpath = os.path.join(datasets, subdir)
#         for filename in os.listdir(subjectpath):
#             path = subjectpath + '/' + filename
#             label = id
#             images.append(cv2.imread(path, 0))
#             labels.append(int(label))
#             #print(labels)
#             print(filename)
#         id += 1
# (width, height) = (130, 100)
#
# (images, labels) = [numpy.array(lis) for lis in [images, labels]]
#
#
# model = cv2.face.LBPHFaceRecognizer_create()
# # model =  cv2.face.FisherFaceRecognizer_create()
# model.train(images, labels)
# model.save("facemodel.pkl")
#
# joblib.dump(names,"names.pkl")
#
# print("over")
import joblib
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import os
# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows = 224
img_cols = 224


num_classes = len(next(os.walk('media/train'))[1])
print(num_classes)
#Loads the VGG16 model
model = VGG16(weights = 'imagenet',
                 include_top = False,
                 input_shape = (img_rows, img_cols, 3))

for layer in model.layers:
    layer.trainable = False

def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "sigmoid")(top_model)
    return top_model



FC_Head = addTopModel(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)



train_data_dir = 'D:\django\project_1\media\\train'
validation_data_dir = 'D:\django\project_1\media\\test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 16
val_batchsize = 8
img_rows = 224
img_cols = 224

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("family_vgg.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# Note we use a very small learning rate
modelnew.compile(loss='binary_crossentropy',
                 optimizer=RMSprop(lr=0.001),
                 metrics=['accuracy'])

nb_train_samples = 160
nb_validation_samples = 40
epochs = 5
batch_size = 16

history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
    )

modelnew.save("family_vgg.h5")
joblib.dump(train_generator.class_indices,"names.pkl")
# print(train_generator.class_indices)