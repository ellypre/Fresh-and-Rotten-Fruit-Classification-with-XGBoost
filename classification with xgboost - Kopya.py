
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load and preprocess data
def load_data(directory):
    images = []
    labels = []
    for folder in os.listdir(directory):
        label = folder.split('_')[0]  # Extract label from folder name
        for file in glob.glob(os.path.join(directory, folder, "*.png")):
            img = cv2.imread(file)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = load_data("C:/HK/yapay zeka/dataset/train")
test_images, test_labels = load_data("C:/HK/yapay zeka/dataset/test")

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Train/test split
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels_encoded, test_size=0.2, random_state=seed)

# Extract features using pretrained VGG16
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x_train_features = base_model.predict(x_train)
x_val_features = base_model.predict(x_val)
x_test_features = base_model.predict(test_images)

base_model.summary()

# Flatten features
x_train_flat = x_train_features.reshape(x_train_features.shape[0], -1)
x_val_flat = x_val_features.reshape(x_val_features.shape[0], -1)
x_test_flat = x_test_features.reshape(x_test_features.shape[0], -1)

# Train XGBoost model
model = XGBClassifier(tree_method='hist', random_state=seed) #işlemi hızlandırmak histogram tabanlı algoritma kullanılır(büyük veri kümeleri üzerinde performansı artırır)
model.fit(x_train_flat, y_train)

# Predictions
train_preds = model.predict(x_train_flat)
val_preds = model.predict(x_val_flat)
test_preds = model.predict(x_test_flat)

# Accuracy
train_accuracy = accuracy_score(y_train, train_preds)
val_accuracy = accuracy_score(y_val, val_preds)
test_accuracy = accuracy_score(test_labels_encoded, test_preds)

print("Train Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)

# Confusion Matrix
cm = confusion_matrix(test_labels_encoded, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("x_train shape:", x_train_flat.shape)
print("x_val shape:", x_val_flat.shape)
print("x_test shape:", x_test_flat.shape)

#----------------------------------xai_lime-------------------------------------

import lime
import lime.lime_tabular


# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    x_train_flat,
    feature_names=[f'feature_{i}' for i in range(x_train_flat.shape[0])],
    class_names=label_encoder.classes_,
    verbose=True,
    mode='classification')

# Pick an instance to explain (let's choose a random test instance)
np.random.seed(seed)
instance_idx = np.random.randint(0, x_test_flat.shape[0])
instance = x_test_flat[instance_idx]

# Explain the model's prediction on the chosen instance
exp = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=15,  # Number of features to include in the explanation
    top_labels=1      # Number of labels to explain
    )

# Print the explanation
print(f"Explanation for instance {instance_idx}:")
print(exp.as_list(label=np.argmax(model.predict_proba([instance]))))

# Visualize the explanation
fig = exp.as_pyplot_figure(label=np.argmax(model.predict_proba([instance])))
plt.show()



#---------------------------------------------------------------------------------

#-----------------------------------SHAP------------------------------------------
import shap
import pandas as pd

# SHAP explainer'ı oluştur
explainer = shap.Explainer(model, x_train_flat)

# SHAP değerlerini hesapla
shap_values = explainer.shap_values(x_train_flat)

#özelliklerin olduğu data frame oluşturma
df = pd.DataFrame(data=x_train_flat, columns=["feature_{}".format(i) for i in range(x_train_flat.shape[1])])

# DataFrame'in sütun isimlerini alın
feature_names = df.columns.tolist()

shap.summary_plot(shap_values, x_train_flat, feature_names=feature_names)

#--------------------------------------------------------------------------------

#----------------------------------learning_curve------------------------------
from sklearn.model_selection import learning_curve

# Define function to plot learning curve
def plot_learning_curve(estimator, X, y):
    
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, train_sizes=np.geomspace(0.001, 1, 5), cv=5,
                                                           scoring='accuracy', n_jobs=1, shuffle=True )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.xscale('log')
    plt.grid()
    plt.show()

# Plot learning curve
plot_learning_curve(model, x_train_flat, y_train)
