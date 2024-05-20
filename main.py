from data_preparation import *
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


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

def define_dataset():
    main_dataset_folder_path = "aclImdb_v1/aclImdb/"
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

def learn_NB_model(X_train, y_train, X_test, y_test, models=[("MultinomialNB", MultinomialNB())]):
    for model_name, model in models:
        print(f"Evaluating {model_name}")
        # GaussianNB requires dense arrays instead of sparse matrices
        if model_name == 'GaussianNB':
            X_train = X_train.toarray()
            X_test = X_test.toarray()

        # Learn model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        # Evaluate accuracy
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Evaluate the classifier
        print(f"Naive Base: Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")
        print("Classification Report:\n", classification_report(test_labels, y_pred_test))
        print("Confusion Matrix:\n", confusion_matrix(test_labels, y_pred_test))


def learn_SVM_model(X_train, y_train, X_test, y_test, kernel_functions = ['linear']):
    # Dictionary to store results
    results = {}
    # Loop over kernel functions
    for kernel in kernel_functions:
        # Initialize SVM classifier
        svm_classifier = SVC(kernel=kernel)
        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)
        # Make predictions
        y_pred_train = svm_classifier.predict(X_train)
        y_pred_test = svm_classifier.predict(X_test)
        # Evaluate accuracy
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        # Store and print results
        print(f"Kernel: {kernel}, Training Accuracy: {train_accuracy}, Testing Accuracy: {test_accuracy}")
        results[kernel] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}
    # Plotting kernel function results
    kernels = list(results.keys())
    train_accuracies = [result['train_accuracy'] for result in results.values()]
    test_accuracies = [result['test_accuracy'] for result in results.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(kernels, train_accuracies, color='b', alpha=0.5, label='Training Accuracy')
    plt.bar(kernels, test_accuracies, color='r', alpha=0.5, label='Testing Accuracy')
    plt.xlabel('Kernel Function')
    plt.ylabel('Accuracy')
    plt.title('SVM Classifier Performance for Different Kernel Functions')
    plt.legend()
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig("svm_learning_kernel_functions_results.png")

if __name__ == '__main__':
    # -------- Define Data --------
    train_reviews, train_labels, test_reviews, test_labels = define_dataset()
    # Combine train and test data for visualization
    all_reviews = train_reviews + test_reviews
    all_labels = train_labels + test_labels

    # -------- Vectorize Data --------
    # Vectorize the text data using TF-IDF with stop words removal
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(tqdm(all_reviews, desc="Vectorizing text data"))

    # -------- Visualize Data --------
    # # Visualize with PCA
    pca_result = visualize_dataset_with_PCA(
        tfidf_matrix.toarray(), n_components=100
    )
    # # Visualize with MDS
    # visualize_dataset_with_MDS(pca_result)
    # # Visualize with LLE
    visualize_dataset_with_LLE(pca_result)

    # -------- Learning model --------
    # Split data into training and testing sets
    X_train = tfidf_matrix[:len(train_reviews)]
    y_train = train_labels
    X_test = tfidf_matrix[len(train_reviews):]
    y_test = test_labels

    # Learn the SVM model
    learn_NB_model(
        X_train, y_train, X_test, y_test,
        models=[
            ('MultinomialNB', MultinomialNB()),
            ('GaussianNB', GaussianNB()),
            ('BernoulliNB', BernoulliNB()),
            ('ComplementNB', ComplementNB())
        ]
    )

    # Learn the SVM model
    learn_SVM_model(
        X_train, y_train, X_test, y_test,
        kernel_functions=[
            'linear',
            'poly',
            'rbf',
            'sigmoid'
        ]
    )





