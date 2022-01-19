#Module Importing

try: 
	from skimage.io import imread, imshow
	from tqdm import tqdm
	import math
	import matplotlib.pyplot as plt
	import numpy
	import os
	import pandas
	import pywt
	import shutil
	import tensorflow
	from tensorflow import keras
	from tensorflow.keras import layers, optimizers
	from tensorflow.keras.models import Sequential, Model
	from sklearn import metrics
	from keras.callbacks import CSVLogger
	from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    
except ImportError:
    print('Please download dependencies in Requirements.txt')


# # Data Preprocessing
# Images are seperated by tumor type

Training_Path = './dataset'
Train_Image_Path = './dataset/image'
Sorted_Training_Data_Path = os.path.join(Training_Path, 'tumor_types')

#Read Train CSV File
Labels_Train = pandas.read_csv(os.path.join(Training_Path, 'label.csv'))

#Make main directory
os.makedirs(Sorted_Training_Data_Path, exist_ok=True)

# Make a directory for images separated into tumor types
for label in Labels_Train['label'].unique():
    os.makedirs(os.path.join(Sorted_Training_Data_Path, label), exist_ok=True)

Already_Sorted = True

if Already_Sorted:
	pass
else:
# Sort Images
	for Image_Number in tqdm(Labels_Train['file_name']):
		label_data = Labels_Train.index[Labels_Train['file_name'] == Image_Number].item()
		if Labels_Train.loc[label_data, 'label'] == 'no_tumor':
			shutil.copy(os.path.join(Train_Image_Path, Image_Number),
						os.path.join(Sorted_Training_Data_Path, 'no_tumor'))
		elif Labels_Train.loc[label_data, 'label'] == 'glioma_tumor':
			shutil.copy(os.path.join(Train_Image_Path, Image_Number),
						os.path.join(Sorted_Training_Data_Path, 'glioma_tumor'))
		elif Labels_Train.loc[label_data, 'label'] == 'meningioma_tumor':
			shutil.copy(os.path.join(Train_Image_Path, Image_Number),
						os.path.join(Sorted_Training_Data_Path, 'meningioma_tumor'))
		elif Labels_Train.loc[label_data, 'label'] == 'pituitary_tumor':
			shutil.copy(os.path.join(Train_Image_Path, Image_Number),
						os.path.join(Sorted_Training_Data_Path, 'pituitary_tumor'))



#Read Images
images = []
for Image_Number in tqdm(Labels_Train['file_name']):
    image_data = imread(os.path.join(Train_Image_Path, Image_Number))
    images.append(image_data)


Test_Path = './test'
Test_Image_Path = './test/image'
Sorted_Test_Path = os.path.join(Test_Path, 'tumor_types')


# Read Test CSV File
Labels_Test = pandas.read_csv(os.path.join(Test_Path, 'label.csv'))

# Split up testing images
for label in Labels_Test['label'].unique():
    os.makedirs(os.path.join(Sorted_Test_Path, label), exist_ok=True)

if Already_Sorted:
	pass
else:
# Move images
	for Image_Number in tqdm(Labels_Test['file_name']):
		label_data = Labels_Test.index[Labels_Test['file_name']==Image_Number].item()
		if Labels_Test.loc[label_data, 'label'] == 'no_tumor':
		  shutil.copy(os.path.join(Test_Image_Path, Image_Number), os.path.join(Sorted_Test_Path, 'no_tumor'))
		elif Labels_Test.loc[label_data, 'label'] =='glioma_tumor':
		  shutil.copy(os.path.join(Test_Image_Path, Image_Number), os.path.join(Sorted_Test_Path, 'glioma_tumor'))
		elif Labels_Test.loc[label_data, 'label'] =='meningioma_tumor':
		  shutil.copy(os.path.join(Test_Image_Path, Image_Number), os.path.join(Sorted_Test_Path, 'meningioma_tumor'))
		elif Labels_Test.loc[label_data, 'label'] =='pituitary_tumor':
		  shutil.copy(os.path.join(Test_Image_Path, Image_Number), os.path.join(Sorted_Test_Path, 'pituitary_tumor'))



# Preprocessing raw image data to a dataset object that can be used to train the model
Training_Dataset = keras.preprocessing.image_dataset_from_directory(Sorted_Training_Data_Path,
  labels='inferred', 
  label_mode='int',
  validation_split=0.2,
  seed=1,
  subset="training",
  shuffle = True,
  image_size=(256, 256))

Validation_Dataset = keras.preprocessing.image_dataset_from_directory(
  Sorted_Training_Data_Path,
  labels='inferred', 
  label_mode='int',
  validation_split=0.2,
  seed=1,
  subset="validation",
  shuffle = True,
  image_size=(256, 256))

Testing_Dataset = keras.preprocessing.image_dataset_from_directory(
  Sorted_Test_Path,
  labels='inferred', 
  label_mode='int',
  shuffle = True,
  image_size=(256, 256))


class_names = Training_Dataset.class_names

# Example labelled images

plt.figure(figsize=(10, 10))
for images, labels in Training_Dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

tensorflow.keras.backend.clear_session()

# Use the Xception architecture for the model
base_model = keras.applications.Xception(input_shape=(256, 256, 3), weights='imagenet', include_top=False)


# Freeze the convolutional base before compile and train the model.
# Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training. 
base_model.trainable = False


Inputs = keras.Input(shape=(256, 256, 3))
Processed_Inputs = keras.applications.xception.preprocess_input(Inputs)
Processed_Inputs = base_model(Processed_Inputs, training=False)
Pool_Layer = layers.GlobalAveragePooling2D()(Processed_Inputs)
Drop_Layer = layers.Dropout(0.2)(Pool_Layer)
Outputs = layers.Dense(4, activation='softmax')(Drop_Layer)
model = Model(Inputs, Outputs)


loss = keras.losses.SparseCategoricalCrossentropy
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=loss(from_logits=False), metrics=['accuracy'])
model.summary()


# Set initial epochs before fine tuning
Starting_Epochs = 30

# Stops early if model does not improve over 3 iterations
Non_Improving = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)


#If saved model already exists, use it (cuts down on processing time when testing)
if os.path.exists('./Model_1'):
	print('Using preexisting model')
	keras.models.load_model('./Model_1')
	log_data_one = pandas.read_csv('training.log', sep=',', engine='python')
# Fits model and logs training data to a CSV file
else:
	csv_logger = CSVLogger('training.log', separator=',', append=False)
	history = model.fit(Training_Dataset,
						epochs=Starting_Epochs,
						validation_data=Validation_Dataset,
						callbacks=[csv_logger])
						
	model.save("Model_1")




#Unfreeze training model
base_model.trainable = True

#Compiles the model
model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()


# Total Epochs (Including initial ones)
Total_Epochs = 40

#If saved model already exists, use it (cuts down on processing time when testing)
if os.path.exists('./Model_2'):
	print('Using preexisting model')
	keras.models.load_model('./Model_2')
	log_data_two = pandas.read_csv('training_2.log', sep=',', engine='python')
	saved = True
# Fits model and logs training data to a CSV file
else:
	csv_logger = CSVLogger('training_2.log', separator=',', append=False)
	history_fine = model.fit(Training_Dataset,
							 epochs=Total_Epochs,
							 initial_epoch=Starting_Epochs,
							 validation_data=Validation_Dataset,
							 callbacks=[csv_logger, Non_Improving])
							 
	model.save('Model_2')
	saved = False

Adds data from both together (Note: Will fail if you have a saved "Model 1" and no saved "Model 2". If so, just run twice)
if saved:
	acc = [*log_data_one['accuracy'], *log_data_two['accuracy']]
	val_acc = [*log_data_one['val_accuracy'], *log_data_two['val_accuracy']]
	loss = [*log_data_one['loss'], *log_data_two['loss']]
	val_loss = [*log_data_one['val_loss'], *log_data_two['val_loss']]
else:
	acc = history.history['accuracy'] + history_fine.history['accuracy']
	val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
	loss = history.history['loss'] + history_fine.history['loss']
	val_loss = history.history['val_loss'] + history_fine.history['val_loss']


plt.style.use('classic')
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([Starting_Epochs-1,Starting_Epochs-1],
          plt.ylim(), label='Fine Tuning Started')
plt.legend(loc='lower right')
plt.title('Accuracy (Training/Validation)')
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot([Starting_Epochs-1,Starting_Epochs-1],
         plt.ylim(), label='Fine Tuning Started')
plt.legend(loc='upper right')
plt.title('Loss (Training/Validation)')
plt.xlabel('Epoch')
plt.show()




predictions = numpy.array([])
labels = numpy.array([])
# Obtain the predictions from test dataset
for x, y in tqdm(Testing_Dataset):
  predictions = numpy.concatenate([predictions, numpy.argmax(model.predict(x), axis=-1)])
  labels = numpy.concatenate([labels, x.numpy()])

display = metrics.ConfusionMatrixDisplay.from_predictions(
        labels,
        predictions,
        display_labels = Testing_Dataset.class_names,
        xticks_rotation = 'vertical',
        cmap=plt.cm.Blues,
        normalize='true'
    )

disp.ax_.grid(False)
display.ax_.set_title('Transfer Learning with Xception')
plt.show()

print(metrics.classification_report(
    labels, predictions, target_names=Testing_Dataset.class_names))

