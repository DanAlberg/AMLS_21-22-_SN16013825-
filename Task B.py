#!/usr/bin/env python
# coding: utf-8

# # Module Importing
# To see requirements, please see Requirements.txt

# In[14]:


try: 
    from skimage.feature import hog
    from skimage.io import imread, imshow
    from skimage.transform import resize
    from sklearn import svm, metrics
    from sklearn.decomposition import PCA
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    import math
    import matplotlib.pyplot as plt
    import numpy
    import os
    import pandas
    import pywt
    
except ImportError:
    print('Please download dependencies in Requirements.txt')


# # Data Preprocessing

# In[2]:


training_dataset = pandas.read_csv('./dataset/label.csv')
training_dataset.head()


# # Preprocessing Training Images
# 
# Using pywavelets, the training images reduced from (512, 512) to (256, 256), which reduces noise.

# In[3]:


training_path = './dataset/image'

train_images = []
for filename in tqdm(training_dataset['file_name']):
    image = imread(os.path.join(training_path, filename), as_gray=True)   # Converts image to greyscale, then gets data
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    train_images.append(cA)


# # Preprocessing Test Images
# 
# Same method is carried out on test images

# In[4]:


testing_dataset = pandas.read_csv('./test/label.csv')
test_path = './test/image'

test_images = []
for filename in tqdm(testing_dataset['file_name']):
    image = imread(os.path.join(test_path, filename,), as_gray=True)   # Converts image to greyscale, then gets data
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    test_images.append(cA)


# # Feature Extraction

# # Histogram of Orientated Gradients (HOG)
# 
# For each inputted image, returns detected HOG features

# In[5]:


def hog_FE(image_data):

    hog_features = []
    
    for i in tqdm(range(len(image_data))):
        resized = resize(image_data[i], (128, 64))
        # Extract Histogram of Oriented Gradients (HOG) for the image
        fd = hog(resized, orientations = 9, pixels_per_cell = (8, 8), cells_per_block = (2, 2), visualize = False, block_norm='L2-Hys')
        hog_features.append(fd)

    return hog_features


# # Convert Label to Binary
# 
# For each image, it checks the "label" status, and assigns it a 0 if there is no tumor, and a 1 if there is

# In[8]:


def label_to_number(dataset):
    tumortype = []
    for label in dataset['label']:
        if 'no_tumor' in label:
            tumortype.append(0)
        elif 'glioma_tumor' in label:
            tumortype.append(1)
        elif 'meningioma_tumor' in label:
            tumortype.append(2)
        elif 'pituitary_tumor' in label:
            tumortype.append(3)
    return tumortype


# # Set Creation

# In[9]:


train = pandas.DataFrame(data=hog_FE(train_images))
train['tumor_binary'] = label_to_number(training_dataset)

test = pandas.DataFrame(data=hog_FE(test_images))
test['tumor_binary'] = label_to_number(testing_dataset)
   
    
    
Training_Dataset = train.drop('tumor_binary', axis=1)  # All features apart from tumor type
Training_Tumor = train['tumor_binary']  # Training images tumor type

Test_Dataset = test.drop('tumor_binary', axis=1)  # All features apart from tumor type
Test_Tumor = test['tumor_binary']  # Test images tumor type

scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
scaler.fit(Training_Dataset) # Fits scalar to the training dataset

Training_Dataset = scaler.transform(Training_Dataset) # Transforms Training_Dataset with scalar
Test_Dataset = scaler.transform(Test_Dataset) # Transforms Test_Dataset with scalar


# In[10]:


pandas.Series(Training_Tumor).value_counts()


# # SVM Classifier

# In[15]:


'''
Inputs
    Training_Dataset: Training dataset;
    Training_Tumor: Training labels;
    Test_Dataset: Testing dataset.

Return
    Y_pred: Predicted labels from Test_Dataset using SVM;
    results: Table of cross-validation results.
'''


pca = PCA()
svc = svm.SVC(probability=True)

pca_svc = Pipeline(steps=[('pca', pca), ('svc', svc)])

params = [{'svc__C': [1, 10, 100], 'svc__kernel': ['rbf'], 'svc__gamma': ['auto'],
           "pca__n_components": [0.96, 0.97, 0.98, 0.99]}, # Parameters for HG-SVC
          {'svc__C': [1, 10, 100], 'svc__kernel': ['poly'], 'svc__gamma': ['auto'],
           "pca__n_components": [0.96, 0.97, 0.98, 0.99]}] # Parameters for HG-SVC

classifier = HalvingGridSearchCV(pca_svc, params, factor = 2, verbose= 3, return_train_score=True)
classifier.fit(Training_Dataset, Training_Tumor)
print(classifier.best_params_)

results = pandas.DataFrame.from_dict(
    classifier.cv_results_).sort_values(by=['rank_test_score'])
results = results[['params', 'mean_train_score',
                   'mean_test_score', 'rank_test_score']]

Tumor_Prediction = classifier.predict(Test_Dataset)


# # Confusion Matrix Plot

# In[18]:


fig, ax = plt.subplots()
metrics.ConfusionMatrixDisplay.from_predictions(
    Test_Tumor,
    Tumor_Prediction,
    cmap=plt.cm.Blues,
    normalize='true',
    ax=ax,
    display_labels=['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor'],
    xticks_rotation = 'vertical'
)
ax.set_title('Task B SVM Classifier')


# # Classification Report

# In[21]:


print(metrics.classification_report(
    Test_Tumor, Tumor_Prediction, target_names=['No tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))
pandas.set_option('display.max_colwidth', None)
display(results)


# In[ ]:




