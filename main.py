import training
import sklearn
# from data_preproccess import create_data_list, create_tfidf
from dataloader import Dataloader
from config import Config

TEST_SIZE = 0.15
RANDOM_STATE = 42


if __name__ == "__main__":
    # words_list, y = create_data_list(input_file)
    config = Config()
    dataloader = Dataloader(config)
    # word1 = create_word2vec(words_list)
    # x_data = create_tfidf(words_list)
    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataloader.all, dataloader.all_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    x_train, y_train = dataloader.get_train_dataloader()
    x_test, y_test = dataloader.get_test_dataloader()

    print("Results with common sarcstic words: ")
    training.bernoulli_model(x_train, x_test, y_train, y_test)
    training.KNN_model(x_train, x_test, y_train, y_test)
    training.SVM_model(x_train, x_test, y_train, y_test)

    x_train_tfidf, y_train_tfidf = dataloader.get_train_dataloader(tfidf=True)
    x_test_tfidf, y_test_tfidf = dataloader.get_test_dataloader(tfidf=True)
    
    print()
    print("Results with common sarcstic words + TF-IDF: ")
    training.bernoulli_model(x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf)
    training.KNN_model(x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf)
    training.SVM_model(x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf)