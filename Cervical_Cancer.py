from tkinter import *
from tkinter import filedialog, ttk
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import numpy as np

# Initialize main tkinter window
main = Tk()
main.title("Optimized Stacked Ensemble Techniques for Cervical Cancer Prediction Using SMOTE and RFERF")
main.geometry("1300x1200")

# Global variables
filename = None
dataset = None
X = None
Y = None
X_train = None
X_test = None
y_train = None
y_test = None
rfe = None
clf = None

def upload():
    """Upload a dataset and display its basic information."""
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset", title="Select a File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    text.delete('1.0', END)
    if filename:
        dataset = pd.read_csv(filename)
        text.insert(END, f"{filename} loaded successfully!\n\n")
        text.insert(END, f"Dataset Size:\nTotal Rows    : {dataset.shape[0]}\n")
        text.insert(END, f"Total Columns : {dataset.shape[1]}\n\n")
        text.insert(END, f"Dataset Samples:\n{dataset.head()}\n\n")
    else:
        text.insert(END, "No file selected.\n")

def preprocess():
    """Preprocess the dataset by handling missing values and displaying class distribution."""
    global dataset
    if dataset is None:
        text.insert(END, "Please upload a dataset first.\n")
        return
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)
    unique, counts = np.unique(dataset['Biopsy'], return_counts=True)
    text.insert(END, "Number of Class Labels Before SMOTE:\n")
    text.insert(END, f"Class Label {unique[0]}: {counts[0]}\n")
    text.insert(END, f"Class Label {unique[1]}: {counts[1]}\n\n")

def smoteBalancing():
    """Balance the dataset using SMOTE."""
    global X, Y, dataset, X_train, X_test, y_train, y_test
    if dataset is None:
        text.insert(END, "Please upload and preprocess the dataset first.\n")
        return
    text.delete('1.0', END)
    Y = dataset['Biopsy'].values
    dataset.drop(['Biopsy'], axis=1, inplace=True)
    X = dataset.values
    sm = SMOTE(random_state=42)
    X, Y = sm.fit_resample(X, Y)
    unique, counts = np.unique(Y, return_counts=True)
    text.insert(END, "Number of Class Labels After SMOTE:\n")
    text.insert(END, f"Class Label {unique[0]}: {counts[0]}\n")
    text.insert(END, f"Class Label {unique[1]}: {counts[1]}\n\n")

def featuresSelection():
    """Select important features using RFE."""
    global X, Y, rfe, X_train, X_test, y_train, y_test
    if X is None or Y is None:
        text.insert(END, "Please perform SMOTE balancing first.\n")
        return
    text.delete('1.0', END)
    text.insert(END, f"Total Features Before RFE: {X.shape[1]}\n")
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=20)
    X = rfe.fit_transform(X, Y)
    text.insert(END, f"Total Features After RFE: {X.shape[1]}\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END, f"Training Records: {X_train.shape[0]}\n")
    text.insert(END, f"Testing Records: {X_test.shape[0]}\n")

def trainStacked():
    """Train a stacked ensemble model."""
    global clf, X_train, X_test, y_train, y_test
    if X_train is None or y_train is None:
        text.insert(END, "Please perform feature selection first.\n")
        return
    text.delete('1.0', END)
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('dt', DecisionTreeClassifier()),
        ('knn', KNeighborsClassifier(n_neighbors=2))
    ]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    p = precision_score(y_test, predictions, average='macro') * 100
    r = recall_score(y_test, predictions, average='macro') * 100
    f = f1_score(y_test, predictions, average='macro') * 100
    a = accuracy_score(y_test, predictions) * 100
    text.insert(END, f"Stacking Ensemble Accuracy : {a:.2f}%\n")
    text.insert(END, f"Stacking Ensemble Precision: {p:.2f}%\n")
    text.insert(END, f"Stacking Ensemble Recall   : {r:.2f}%\n")
    text.insert(END, f"Stacking Ensemble F1 Score: {f:.2f}%\n\n")
    conf_matrix = confusion_matrix(y_test, predictions)
    sns.heatmap(conf_matrix, annot=True, cmap="viridis", fmt="g")
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()

def predict():
    """Make predictions using the trained model on test data."""
    global clf, rfe
    if clf is None or rfe is None:
        text.insert(END, "Please train the model first.\n")
        return
    testfile = filedialog.askopenfilename(initialdir="Dataset", title="Select Test Data", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not testfile:
        text.insert(END, "No test file selected.\n")
        return
    dataset = pd.read_csv(testfile)
    dataset.fillna(0, inplace=True)
    dataset = rfe.transform(dataset.values)
    predictions = clf.predict(dataset)
    for i, prediction in enumerate(predictions):
        result = "NORMAL" if prediction == 0 else "CERVICAL CANCER"
        text.insert(END, f"Test Data {i+1}: Predicted as {result}\n")

def close():
    """Close the application."""
    main.destroy()

# GUI Design
font = ('times', 16, 'bold')
title = Label(main, text="Optimized Stacked Ensemble Techniques for Cervical Cancer Prediction Using SMOTE and RFERF")
title.config(bg='LightGoldenrod1', fg='medium orchid', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=100)
text.place(x=400, y=100)
text.config(font=font1)

uploadButton = Button(main, text="Upload Cervical Cancer Dataset", command=upload, font=font1)
uploadButton.place(x=50, y=100)

processButton = Button(main, text="Preprocess Dataset", command=preprocess, font=font1)
processButton.place(x=50, y=150)

smoteButton = Button(main, text="Data Balancing Using SMOTE", command=smoteBalancing, font=font1)
smoteButton.place(x=50, y=200)

featuresButton = Button(main, text="Features Selection Using RFERF", command=featuresSelection, font=font1)
featuresButton.place(x=50, y=250)

stackedButton = Button(main, text="Train Stacked Ensemble Algorithm", command=trainStacked, font=font1)
stackedButton.place(x=50, y=300)

predictButton = Button(main, text="Predict Cancer from Test Data", command=predict, font=font1)
predictButton.place(x=50, y=350)

exitButton = Button(main, text="Exit", command=close, font=font1)
exitButton.place(x=50, y=400)

main.config(bg='OliveDrab2')
main.mainloop()
