# Introduction
This repositary contains three projects that can be used to identify the presence and type of tumour from MRI brain scans. 
The first two implementations use a SVM based approach with HOG feature detection, and produced an accuracy of 96% and 94% respectively.
The first, using a CNN, showed an accuracy of 91%


# Files
README.md - This readme
Requirements.txt - Python package requirement
Task A.ipynb - Jupyter Notebook for Binary Classification (SVM)
Task B.ipynb - Jupyter Notebook for Multiclass Classification (SVM)
Task B 2.py - Python file for Multiclass Classification (CNN)


# Instructions
1. Ensure Python is installed on device
2. Clone this repository to device
3. Download training and test datasets from these URLS:
http://shorturl.at/hquDP
https://drive.google.com/file/d/1LS_C_4_iOeqOyEoWPPoksrk8lqdBKagB/view
5.  Install requirements with the command: pip install -r Requirements.txt
6.  Extract both files, and place the folders "dataset" and "test" in same folder as ipynb/py files
7.  Run A/B in Jupyter by using the command "Jupyter Notebook", or B2 with the command "python Task_B_2.py"


# Requirements
ipywidgets
matplotlib
numpy
pandas
pywavelets
scikit-image
scikit-learn
scipy
tqdm
tensorflow
joblib
