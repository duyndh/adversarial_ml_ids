__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Johnson, Will",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

from contextlib import redirect_stdout
def log(obj, new_line=True):
    with open("TestResults\\log_" + dataset_name + ".txt", "a") as file_log:
        with redirect_stdout(file_log):
            if new_line:
                print(obj)
            else:
                print(obj, end=" ")

def log2(obj1, obj2):
    log(obj1, new_line=False)
    log(obj2)

# Generate a multilayer perceptron  model or ANN

def mlp_model(X, Y):
    
    # Initializing the ANN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(units = round(X.shape[1]/2), kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
    
    # Adding the second hidden layer
    model.add(Dense(units = round(X.shape[1]/2), kernel_initializer = 'uniform', activation = 'relu'))

    
    if(len(np.unique(Y)) > 2): # Multi-classification task
        # Adding the output layer
        model.add(Dense(units = len(np.unique(Y)), kernel_initializer = 'uniform', activation = 'softmax'))
        # Compiling the ANN
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    else: # Binary classification task
        # Adding the output layer
        model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        # Compiling the ANN
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    log(model.summary())
    
    return model


# Train the multilayer perceptron  model or ANN
def mlp_model_train(X, Y, val_split, batch_size, epochs_count):
    # Callback to stop if validation loss does not decrease
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]

    # Fitting the ANN to the Training set
    history = model.fit(X, Y,
                   callbacks=callbacks,
                   validation_split=val_split,
                   batch_size = batch_size,
                   epochs = epochs_count,
                   shuffle=True)

    log(history.history)
    log(model.summary())
    return history


# Evaluate the multilayer perceptron  model or ANN during test time
def mlp_model_eval(X, Y, history, flag):
    # Predicting the results given instances X
    Y_pred = model.predict_classes(X)

    Y_pred = [1 if y > 0.5 else 0 for y in Y_pred > 0.5]
    Y = [1 if y > 0.5 else 0 for y in Y > 0.5]

    # Breakdown of statistical measure based on classes
    log(classification_report(Y, Y_pred, digits=4, zero_division=0))

    # Making the cufusion Matrix
    cm = confusion_matrix(Y, Y_pred)
    log2("Confusion Matrix:\n", cm)
    log2("Accuracy: ", accuracy_score(Y, Y_pred))

    if(len(np.unique(Y))) == 2:
        log2("F1: ", f1_score(Y, Y_pred, average='binary'))
        log2("Precison: ", precision_score(Y, Y_pred, average='binary'))
        log2("Recall: ", recall_score(Y, Y_pred, average='binary'))
    else:
        f1_scores = f1_score(Y, Y_pred, average=None)
        log2("F1: ", np.mean(f1_scores))
        precision_scores = precision_score(Y, Y_pred, average=None)
        log2("Precison: ", np.mean(precision_scores))
        recall_scores = recall_score(Y, Y_pred, average=None)
        log2("Recall: ", np.mean(recall_scores))

    # ------------ Print Accuracy over Epoch --------------------

    # Intilization of the figure
    myFig = plt.figure(figsize=[12,10])

    plt.plot(history.history['accuracy'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
    plt.plot(history.history['val_accuracy'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
    plt.title('Accuracy over Epoch', fontsize=20, weight='bold')
    plt.ylabel('Accuracy', fontsize=18, weight='bold')
    plt.xlabel('Epoch', fontsize=18, weight='bold')
    plt.legend(['Train', 'Validation'], loc='lower right', fontsize=14)
    plt.xticks(ticks=range(0, len(history.history['accuracy'])))
    
    plt.yticks(fontsize=16)
    #plt.show()
        
    fileName = ''
    if(len(np.unique(Y))) == 2:
        if(flag == 1): #Regular
            fileName = dataset_name + '_ANN_Binary_Classification_Accuracy_over_Epoch_Regular.png'
        else: #Adversarial
            pass
            #fileName = dataset_name + '_ANN_Binary_Classification_Accuracy_over_Epoch_Adversarial.png'
    else:
        if(flag == 1): #Regular
            fileName = dataset_name + '_ANN_Multiclass_Classification_Accuracy_over_Epoch_Regular.png'
        else: #Adversarial
            pass
            #fileName = dataset_name + '_ANN_Multiclass_Classification_Accuracy_over_Epoch_Adversarial.png'
    
    # Saving the figure
    if len(fileName) > 0:
        myFig.savefig("TestResults\\" + fileName, format='png', dpi=1200)
    
    # ------------ Print Loss over Epoch --------------------

    # Clear figure
    plt.clf()
    myFig = plt.figure(figsize=[12,10])
    
    plt.plot(history.history['loss'], linestyle = ':',lw = 2, alpha=0.8, color = 'black')
    plt.plot(history.history['val_loss'], linestyle = '--',lw = 2, alpha=0.8, color = 'black')
    plt.title('Loss over Epoch', fontsize=20, weight='bold')
    plt.ylabel('Loss', fontsize=18, weight='bold')
    plt.xlabel('Epoch', fontsize=18, weight='bold')
    plt.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
    plt.xticks(ticks=range(0, len(history.history['loss'])))
    
    plt.yticks(fontsize=16)
    #plt.show()
        
    fileName = ''
    if(len(np.unique(Y))) == 2:
        if(flag == 1): #Regular
            fileName = dataset_name + '_ANN_Binary_Classification_Loss_over_Epoch_Regular.png'
        else: #Adversarial 
            pass
            #fileName = dataset_name + '_ANN_Binary_Classification_Loss_over_Epoch_Adversarial.png'
    else:
        if(flag == 1): #Regular
            fileName = dataset_name + '_ANN_Multiclass_Classification_Loss_over_Epoch_Regular.png'
        else: #Adversarial
            pass
            #fileName = dataset_name + '_ANN_Multiclass_Classification_Loss_over_Epoch_Adversarial.png'
    
    # Saving the figure
    if len(fileName) > 0:
        myFig.savefig("TestResults\\" + fileName, format='png', dpi=1200)
    
    
    # ------------ ROC Curve --------------------

    # Clear figure
    plt.clf()
    myFig = plt.figure(figsize=[12,10])
    
    if len(np.unique(Y)) == 2:
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        plt.plot(fpr, tpr, color='black',
                label=r'ROC (AUC = %0.3f)' % (auc(fpr, tpr)),
                lw=2, alpha=0.8)
            
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18, weight='bold')
        plt.ylabel('True Positive Rate', fontsize=18, weight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20, fontweight='bold')
        plt.legend(loc="lower right",fontsize=14)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        #plt.show()
        
        fileName = ''
        if(flag == 1): #Regular
            fileName = dataset_name + '_ANN_Binary_Classification_ROC_Regular.png'
        else: #Adversarial
            fileName = dataset_name + '_ANN_Binary_Classification_ROC_Adversarial.png'

        # Saving the figure
        if len(fileName) > 0:
            myFig.savefig("TestResults\\" + fileName, format='png', dpi=1200)


# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# importing cleverhans - an adversarial example library
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks_tf import jacobian_graph
#from cleverhans.attacks import FastGradientMethod
#from cleverhans.utils_tf import model_train, model_eval, batch_eval

# Libraries relevant to performance metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import os

if not os.path.exists("TestResults"):
    os.makedirs("TestResults")

use_CICIDS2017_dataset = False
dataset_name = ""

#importing the data set

# ==== Data processing for CICIDS 2017 ====
if use_CICIDS2017_dataset:
    dataset_name = 'CICIDS2017'
    dataset = pd.read_csv('CICIDS2017_dataset.csv')
    log("================================")
    log(dataset.head())
    log(dataset.shape)

    # Some manual processing on the dataframe
    dataset = dataset.dropna()
    dataset = dataset.drop(['Flow_ID', '_Source_IP', '_Destination_IP', '_Timestamp'], axis = 1, errors= 'ignore')
    dataset['Flow_Bytes/s'] = dataset['Flow_Bytes/s'].astype(float)
    dataset['_Flow_Packets/s'] = dataset['_Flow_Packets/s'].astype(float)

    # Creating X and Y from the dataset
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(dataset['Label'])
    Y_attack = le.transform(dataset['Label'])
    log(list(le.classes_))
    log(np.unique(Y_attack))
    Y_class = dataset.iloc[:,-1].values
    X = dataset.iloc[:,0:80].values
    X = X.astype(int)

else:
    # ==== Data processing for TRAbID 2017 ====
    from scipy.io import arff
    dataset_name = 'TRAbID2017'
    data = arff.loadarff('TRAbID2017_dataset.arff')
    dataset = pd.DataFrame(data[0])
    log("================================")
    log(dataset.head())
    log(dataset.shape)

    # Creating X and Y from the dataset
    X = dataset.iloc[:,0:43].values
    Y_class = pd.read_csv('TRAbID2017_dataset_Y_class.csv')
    Y_class = Y_class.iloc[:,:].values

    Y_class = (1 - Y_class)

# Performing scale data
scaler = MinMaxScaler().fit(X)
X_scaled = np.array(scaler.transform(X))

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_class, test_size = 0.2, random_state = 42, stratify=Y_class)

log("Data Processing has been performed.")

# Tensorflow  placeholder  variables
X_placeholder = tf.placeholder(tf.float32 , shape=(None , X_train.shape[1]))
Y_placeholder = tf.placeholder(tf.float32 , shape=(None))

tf.set_random_seed(42)
model = mlp_model(X_train, Y_train)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

predictions = model(X_placeholder)
log2('Prediction: ', predictions)

# ============== Training the model ==============
history = mlp_model_train(X_train, Y_train,
                0.1, # Validation Split
                64, # Batch Size
                100 # Epoch Count
                )

# ============== Evaluation of the model with actual instances ==============

log("Performance when using actual testing instances")
mlp_model_eval(X_test, Y_test, history, 1)


# ============== Generate adversarial samples for all test datapoints ==============

source_samples = X_test.shape[0]

# Jacobian-based Saliency Map
results = np.zeros((1, source_samples), dtype=float)
perturbations = np.zeros((1, source_samples), dtype=float)
grads = jacobian_graph(predictions , X_placeholder, 1)

X_adv = np.zeros((source_samples, X_test.shape[1]))

for sample_ind in range(0, source_samples):
    # We want to find an  adversarial  example  for  each  possible  target  class
    # (i.e. all  classes  that  differ  from  the  label  given  in the  dataset)
    current_class = int(np.argmax(Y_test[sample_ind]))
    
    # Target the benign class
    for target in [0]:
        if (current_class == 0):
            break
        
        # This call runs the Jacobian-based saliency map approac
        adv_x , res , percent_perturb = SaliencyMapMethod(sess, X_placeholder, predictions , grads,
                                             X_test[sample_ind: (sample_ind+1)],
                                             target , theta=1, gamma =0.1,
                                             increase=True ,
                                             clip_min=0, clip_max=1)
        
        X_adv[sample_ind] = adv_x
        results[target , sample_ind] = res
        perturbations[target , sample_ind] = percent_perturb


# ============== Evaluation of the model with adversarial instances ==============

log("Performance when using adversarial testing instances")
mlp_model_eval(X_adv, Y_test, history, 2)