''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
from matplotlib.pyplot import autoscale

from sklearn.metrics.pairwise import kernel_metrics
warnings.warn = warn

import get_images
import get_landmarks
import numpy as np

''' Import classifier '''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# NB, SVM, ANN

''' Matching and Decision '''
KNN_classifiers = [
    KNeighborsClassifier(n_neighbors=10, p=2),
    KNeighborsClassifier(n_neighbors=5, p=2),
    KNeighborsClassifier(n_neighbors=3, p=2),
    KNeighborsClassifier(n_neighbors=1, p=2),
    KNeighborsClassifier(n_neighbors=1, p=1)
    ]

SVM_classifiers = [
    SVC(),
    make_pipeline(StandardScaler(), SVC()),
    make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
    make_pipeline(StandardScaler(), SVC(class_weight="balanced", C=.1)),
    make_pipeline(StandardScaler(), SVC(class_weight="balanced", gamma=.0001)),
]

landmarks_values = [5, 68]

def run_classifiers(landmarks, dataset, classifiers):

    results = open(dataset + str(" ") + classifiers[0].__class__.__name__ + str(" results.txt"), "w")

    for landmarks in landmarks_values:
        for clf in classifiers: 

            ''' Load the data and their labels '''
            image_directory = dataset
            X, y = get_images.get_images(image_directory)

            ''' Get distances between face landmarks in the images '''
            # get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
            X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', landmarks, False)

            num_correct = 0
            labels_correct = []
            num_incorrect = 0
            labels_incorrect = []

            for i in range(0, len(y)):
                query_img = X[i, :]
                query_label = y[i]
                
                template_imgs = np.delete(X, i, 0)
                template_labels = np.delete(y, i)
                    
                # Set the appropriate labels
                # 1 is genuine, 0 is impostor
                y_hat = np.zeros(len(template_labels))
                y_hat[template_labels == query_label] = 1 
                y_hat[template_labels != query_label] = 0
                
                clf.fit(template_imgs, y_hat) # Train the classifier
                y_pred = clf.predict(query_img.reshape(1,-1)) # Predict the label of the query
                
                # Gather results
                if y_pred == 1:
                    num_correct += 1
                    labels_correct.append(query_label)
                else:
                    num_incorrect += 1
                    labels_incorrect.append(query_label)

            result = ("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f, Landmarks = %d" 
            % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect), landmarks))

            parameters = str(clf.get_params(deep=True))

            classifier_name = clf.__class__.__name__

            # Print results to console and into the text files
            print()
            print(classifier_name)
            print(result)
            results.write(classifier_name + "\n")
            results.write(parameters + "\n")
            results.write(result + "\n\n")

    results.close()

run_classifiers(landmarks_values, "Caltech Faces Dataset", KNN_classifiers)
run_classifiers(landmarks_values, "SoF Dataset", KNN_classifiers)
run_classifiers(landmarks_values, "Caltech Faces Dataset", SVM_classifiers)
run_classifiers(landmarks_values, "SoF Dataset", SVM_classifiers)