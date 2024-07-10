from data_preparation import *
from spinner import *
from terminal_utils import *

import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from alive_progress import alive_bar
from threading import Thread
import time
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
import ssl

import tensorflow_hub as hub
import kagglehub
from transformers import BertTokenizer, BertModel
import torch
import warnings

global LOG_COLOR
LOG_COLOR = '\033[38;2;247;84;100m'

def run_with_spinner(func, func_args=(), func_kwargs={}, description="Processing", color: Fore = LOG_COLOR):
    with Spinner(description=description, color=color):
        # Run the given function with the provided arguments
        result = func(*func_args, **func_kwargs)
    return result

def dataset_visualization_2d(data, save_as, figsize=(10,8), title='Dataset Visualization', xlabel='x', ylabel='y', save_format="png"):
    plt.figure(figsize=figsize)
    colors = ['red' if label == 0 else 'blue' for label in all_labels]
    plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{save_as}.{save_format}")

def visualize_dataset_with_PCA(X, n_components=2):
    # Reduce dimensions using PCA
    pca = PCA(n_components=n_components)
    # pca_result = pca.fit_transform(tqdm(tfidf_matrix.toarray(), desc="Applying PCA"))
    pca_result = pca.fit_transform(X)

    # Visualize PCA results
    dataset_visualization_2d(
        data=pca_result,
        save_as="dataset_pca",
        title="PCA of Movie Reviews TF-IDF",
        xlabel="Principal Component 1",
        ylabel="Principal Component 2"
    )
    return pca_result

def visualize_dataset_with_MDS(X):
    # Reduce dimensions using MDS
    mds = MDS(n_components=2, verbose=1, n_jobs=-1)
    mds_result = mds.fit_transform(X)

    # Visualize MDS results
    dataset_visualization_2d(
        data=mds_result,
        save_as="dataset_mds",
        title="MDS of Movie Reviews TF-IDF",
        xlabel="MDS Component 1",
        ylabel="MDS Component 2"
    )
    return mds_result

def visualize_dataset_with_LLE(X):
    # Reduce dimensions using Locally Linear Embedding (LLE)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=12, n_jobs=-1)
    # lle_result = lle.fit_transform(tqdm(tfidf_matrix.toarray(), desc="Applying LLE"))
    lle_result = lle.fit_transform(X)

    # Visualize PCA results
    dataset_visualization_2d(
        data=lle_result,
        save_as="dataset_lle",
        title="LLE of Movie Reviews TF-IDF",
        xlabel="LLE Component 1",
        ylabel="LLE Component 2"
    )
    return lle_result

def define_dataset(main_dataset_folder_path = "aclImdb/"):
    # Load train data
    train_neg = read_txt_files(folder_path=f"{main_dataset_folder_path}train/neg")
    train_pos = read_txt_files(folder_path=f"{main_dataset_folder_path}train/pos")
    # Load test data
    test_neg = read_txt_files(folder_path=f"{main_dataset_folder_path}test/neg")
    test_pos = read_txt_files(folder_path=f"{main_dataset_folder_path}test/pos")

    # Create lists of reviews and labels
    train_reviews = list(train_neg.values()) + list(train_pos.values())
    train_labels = [0] * len(train_neg) + [1] * len(train_pos)
    test_reviews = list(test_neg.values()) + list(test_pos.values())
    test_labels = [0] * len(test_neg) + [1] * len(test_pos)

    return train_reviews, train_labels, test_reviews, test_labels


def plot_confusion_matrix(cm, classes, save_as, save_format="png", title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{save_as}.{save_format}")


def learn_NB_model(X_train, y_train, X_test, y_test, models=[("MultinomialNB", MultinomialNB())], color:str = Fore.WHITE, data_info=""):
    # Dictionary to store results
    results = {}
    # Loop over kernel models
    for model_name, model in models:
        print_colored_text(f"Evaluating {model_name}", color=color)
        # GaussianNB requires dense arrays instead of sparse matrices
        if model_name == 'GaussianNB':
            X_train = X_train.toarray()
            X_test = X_test.toarray()

        # Learn model
        model.fit(X_train, y_train)

        # Evaluate the classifier
        train_test_accuracy = print_and_plot_model_results(
            model=model,
            model_type_str="nb",
            model_info_heading='Naive Base',
            model_info_str=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            color=color,
            data_info=data_info
        )
        # Save train_test_accuracy
        results[model_name] = train_test_accuracy
    # Plotting kernel function results
    train_test_accuracy_plot(
        data=results,
        title='Naive Base Classifiers Performance',
        xlabel='Classifier',
        save_as=f'nb_classifiers_performance_not_standardized{data_info}'
    )

def learn_SVM_model(X_train, y_train, X_test, y_test, kernel_functions=['linear'], color: str = Fore.WHITE, data_info=""):
    # Dictionary to store results
    results = {}
    # Loop over kernel functions
    for kernel in kernel_functions:
        print_colored_text(f"Evaluating SVM model with kernel {kernel}", color=color)
        # Initialize SVM classifier
        svm_classifier = SVC(kernel=kernel)
        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)

        # Evaluate the classifier
        train_test_accuracy = print_and_plot_model_results(
            model=svm_classifier,
            model_type_str="svm",
            model_info_heading='Kernel',
            model_info_str=kernel,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            color=color,
            data_info=data_info
        )
        # Save train_test_accuracy
        results[kernel] = train_test_accuracy
    # Plotting kernel function results
    train_test_accuracy_plot(
        data=results,
        title='SVM Classifier Performance for Different Kernel Functions',
        xlabel='Kernel Function',
        save_as=f'svm_learning_kernel_functions_performance_not_standardized{data_info}'
    )


def print_and_plot_model_results(model, model_type_str, model_info_heading, model_info_str, X_train, y_train, X_test, y_test, color:str = Fore.WHITE, data_info=""):
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Evaluate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Evaluate the classifier
    print_colored_text(
        f"{model_info_heading}: {model_info_str} | Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}",
        color=color)
    train_test_accuracy = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}

    # Get the classification report and confusion matrix
    report = classification_report(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    print_colored_text(f"Classification Report:\n {report}", color=color)
    print_colored_text(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred_test)}", color=color)
    plot_confusion_matrix(cm, classes=[0, 1],
                          save_as=f"{model_type_str}_classifiers_{model_info_str.lower()}_confusion_metrix{data_info}")
    return train_test_accuracy


def train_test_accuracy_plot(data, title, xlabel, save_as, save_format='png'):
    kernels = list(data.keys())
    train_accuracies = [result['train_accuracy'] for result in data.values()]
    test_accuracies = [result['test_accuracy'] for result in data.values()]

    bar_width = 0.35  # Width of the bars
    index = range(len(kernels))  # The x locations for the groups

    plt.figure(figsize=(10, 6))

    # Plotting training and testing accuracy bars with an offset
    bars_train = plt.bar(index, train_accuracies, bar_width, color='b', alpha=0.5, label='Training Accuracy')
    bars_test = plt.bar([i + bar_width for i in index], test_accuracies, bar_width, color='r', alpha=0.5,
                        label='Testing Accuracy')

    # Adding text annotations on the bars
    for bar in bars_train:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

    for bar in bars_test:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), ha='center', va='bottom')

    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks([i + bar_width / 2 for i in index], kernels, rotation=45)  # Aligning x-ticks with the bars
    plt.legend(loc='lower right')  # Moving the legend to the bottom right
    plt.tight_layout()
    plt.savefig(f"{save_as}.{save_format}")


def download_nltk_stopwords():
    # Workaround for SSL certificate issues when downloading NLTK data
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')


def define_word_tokens_from_sentence_data(data):
    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    def preprocess(text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    return [preprocess(sentence) for sentence in data]


def vectorize_sentences_with_tokens(sentences_tokens, model):
    def vectorize_sentence(sentence_tokens, model):
        # Define the vectors for words in the set of the tokens
        vectors = [model.wv[word] for word in sentence_tokens if word in model.wv]
        # Define sentence as average vector from all words vectors
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # Vectorize sentences tokens
    return np.array([vectorize_sentence(tokens, model) for tokens in sentences_tokens])


def train_word_to_vec_model(X_train, X_test, sg: bool = False):
    # Ensure nltk stopwords is downloaded
    download_nltk_stopwords()
    # Tokenize the reviews
    train_tokens = define_word_tokens_from_sentence_data(X_train)
    test_tokens = define_word_tokens_from_sentence_data(X_test)
    all_tokens = train_tokens + test_tokens
    # Train a Word2Vec model
    w2v_model = Word2Vec(
        sentences=all_tokens,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1 if sg else 0  # 1 fo Skip-Gram and 0 for CBoW
    )
    # Vectorize sentences tokens
    X_train = vectorize_sentences_with_tokens(
        sentences_tokens=train_tokens,
        model=w2v_model
    )
    X_test = vectorize_sentences_with_tokens(
        sentences_tokens=test_tokens,
        model=w2v_model
    )
    return X_train, X_test



def download_google_universal_sentence_encoder():
    # Download latest version
    path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder")
    print_colored_text(f"Path to model files: {path}", color=LOG_COLOR)
    return path

# Optional: Suppress specific FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')


def encode_sentences_in_batches(sentences, model, tokenizer, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Vectorizing Data with BERT"):
        batch_reviews = sentences[i:i + batch_size]
        inputs = tokenizer(batch_reviews, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)


def to_one_hot(y_pred):
    y_pred = np.squeeze(y_pred)
    max_index = np.argmax(y_pred)
    one_hot = np.zeros_like(y_pred)
    one_hot[max_index] = 1
    return one_hot

def visualize_data_with_Xs_ys_data(X_train, X_test, method='pca', n_components=2, n_neighbors=10):
    """
    Visualizes the given train and test data using PCA or LLE.

    Parameters:
    X_train (np.array): Training data.
    X_test (np.array): Testing data.
    method (str): Method to use for dimensionality reduction ('pca' or 'lle').
    n_components (int): Number of components to use for dimensionality reduction.
    n_neighbors (int): Number of neighbors to consider for LLE (only used if method='lle').
    """
    # Combine the datasets for dimensionality reduction
    X_combined = np.vstack((X_train, X_test))

    # Apply the selected dimensionality reduction method
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_transformed = reducer.fit_transform(X_combined)
    elif method == 'lle':
        reducer = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
        X_transformed = reducer.fit_transform(X_combined)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'lle'.")

    # Separate the transformed data back into training and testing sets
    X_train_transformed = X_transformed[:X_train.shape[0]]
    X_test_transformed = X_transformed[X_train.shape[0]:]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], label='Train', alpha=0.6)
    plt.scatter(X_test_transformed[:, 0], X_test_transformed[:, 1], label='Test', alpha=0.6)
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.title(f'{method.upper()} Plot of Train and Test Data')
    plt.legend()
    plt.show()



def dataset_values_plot_visualization_helper(train_data, test_data, x_label, y_label, title, filename, save_format="png", palette='viridis',
                                             alpha=0.5):
    # Create DataFrames for plotting
    train_df = pd.DataFrame(data=train_data, columns=[x_label, y_label, 'label'])
    test_df = pd.DataFrame(data=test_data, columns=[x_label, y_label, 'label'])

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot training data
    sns.scatterplot(data=train_df, x=x_label, y=y_label, hue='label', style='label', markers=['o'], s=100,
                    palette=palette, edgecolor='k', alpha=alpha, legend='full')

    # Plot testing data
    sns.scatterplot(data=test_df, x=x_label, y=y_label, hue='label', style='label', markers=['X'], s=100,
                    palette=palette, edgecolor='k', alpha=alpha, legend=False)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Label')
    plt.grid(True)
    plt.savefig(f"{filename}.{save_format}")
    # plt.show()


def visualize_dataset_values_with_PCA(X_train, X_test, y_train, y_test, filename, save_format="png", palette='viridis', alpha=0.5):
    # Perform PCA
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Append labels for plotting
    train_data = np.column_stack((X_train_pca, y_train))
    test_data = np.column_stack((X_test_pca, y_test))

    # Plot using the helper function
    dataset_values_plot_visualization_helper(train_data, test_data, 'PC1', 'PC2', 'PCA Plot', filename, save_format, palette, alpha)


def visualize_dataset_values_with_LLE(X_train, X_test, y_train, y_test, filename, save_format="png", palette='viridis', n_neighbors=10,
                                      n_components=2):
    # Perform LLE
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
    X_train_lle = lle.fit_transform(X_train)
    X_test_lle = lle.transform(X_test)

    # Append labels for plotting
    train_data = np.column_stack((X_train_lle, y_train))
    test_data = np.column_stack((X_test_lle, y_test))

    # Plot using the helper function
    dataset_values_plot_visualization_helper(train_data, test_data, 'Component 1', 'Component 2', 'LLE Plot', filename, save_format, palette)

def plot_nn_performance(history, filename):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.savefig(f"{filename}.png")


if __name__ == '__main__':
    # -------- Define Data --------
    X_train, y_train, X_test, y_test = define_dataset()
    # Combine train and test data for visualization
    all_reviews = X_train + X_test
    all_labels = y_train + y_test

    # -------- Vectorize Data --------
    # # Vectorize the text data using TF-IDF with stop words removal
    # vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    # tfidf_matrix = vectorizer.fit_transform(tqdm(all_reviews, desc="Vectorizing text data"))
    # # Split data into training and testing sets
    # X_train = tfidf_matrix[:len(X_train)]
    # X_test = tfidf_matrix[len(X_test):]

    # # Standardize the data
    # scaler = StandardScaler(with_mean=False)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # # Vectorize with CBoW
    # X_train, X_test = train_word_to_vec_model(
    #     X_train,
    #     X_test,
    #     sg=False
    # )

    # # Vectorize with Skip-gram
    # X_train, X_test = train_word_to_vec_model(
    #     X_train,
    #     X_test,
    #     sg=True
    # )

    # # # Load the Universal Sentence Encoder
    # path = download_google_universal_sentence_encoder()
    # embed = hub.load(path)
    # # # Encode the reviews
    # X_train = embed(X_train)
    # X_test = embed(X_test)

    # # Load pre-trained BERT model and tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 'distilbert-base-uncased'
    # model = BertModel.from_pretrained('bert-base-uncased')  # 'distilbert-base-uncased'
    # # Encode the reviews
    # X_train = encode_sentences_in_batches(X_train, model, tokenizer)
    # X_test = encode_sentences_in_batches(X_test, model, tokenizer)
    # # Save the vectorized data
    # np.save('train_vectors.npy', X_train)
    # np.save('test_vectors.npy', X_test)
    # # Load the vectorized data
    X_train = np.load('train_vectors.npy')
    X_test = np.load('test_vectors.npy')

    # -------- Visualize Data --------
    # # Visualize with PCA TF-IDF
    # pca_result = run_with_spinner(
    #     visualize_dataset_with_PCA,
    #     func_args=(tfidf_matrix.toarray(), ),
    #     func_kwargs={'n_components': 2},
    #     description="PCA and Visualizing Dataset",
    #     color=LOG_COLOR
    # )
    # # Visualize with PCA BERT
    # run_with_spinner(
    #     visualize_dataset_values_with_PCA,
    #     func_args=(X_train, X_test, y_train, y_test),
    #     func_kwargs={'filename': "dataset_pca_bert"},
    #     description="PCA and Visualizing Dataset",
    #     color=LOG_COLOR
    # )
    # # Visualize with MDS
    # run_with_spinner(
    #     visualize_dataset_with_MDS,
    #     func_args=(pca_result, ),
    #     description="MDS and Visualizing Dataset",
    #     color=LOG_COLOR
    # )
    # # Visualize with LLE
    # run_with_spinner(
    #     visualize_dataset_with_LLE,
    #     func_args=(pca_result, ),
    #     description="LLA and Visualizing Dataset",
    #     color=LOG_COLOR
    # )
    # # # Visualize with LLE BERT
    # run_with_spinner(
    #     visualize_dataset_values_with_LLE,
    #     func_args=(X_train, X_test, y_train, y_test),
    #     func_kwargs={'filename': "dataset_lle_bert"},
    #     description="LLE and Visualizing Dataset",
    #     color=LOG_COLOR
    # )

    # -------- Learning model --------

    # # Learn the NB model
    # learn_NB_model(
    #     X_train, y_train, X_test, y_test,
    #     models=[
    #         ('MultinomialNB', MultinomialNB()),
    #         ('GaussianNB', GaussianNB()),
    #         ('BernoulliNB', BernoulliNB()),
    #         ('ComplementNB', ComplementNB())
    #     ],
    #     data_info="_tfidf_data"
    # )

    # # Learn the SVM model
    # learn_SVM_model(
    #     X_train, y_train, X_test, y_test,
    #     kernel_functions=[
    #         'linear',
    #         'poly',
    #         'rbf',
    #         'sigmoid'
    #     ],
    #     data_info="_bert_data"
    # )

    # -------- Learning NN --------
    # Ensure that the input data is a numpy array and has the correct shape
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Reshape the input data to add a dummy dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print("Reshaped X_train shape:", X_train.shape)
    print("Reshaped X_test shape:", X_test.shape)

    # Define the input shape based on BERT embeddings shape
    input_shape = (X_train.shape[1], 1)  # Adding a dummy dimension

    # # Build the NN model
    model = Sequential([
        Flatten(input_shape=(768, 1)),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(25, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=225, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

    # Predict the labels for the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Calculate and print the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score: {accuracy:.4f}')

    # Plot performance
    plot_nn_performance(history, filename="nn_training_performance_bert_data")
