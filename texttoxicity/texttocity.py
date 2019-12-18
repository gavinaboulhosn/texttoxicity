from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from base import *
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")
plt.interactive(False)

param_grid = dict(
    num_filters=[32, 64, 128],
    kernel_size=[3, 5, 7],
    vocab_size=[5000],
    embedding_dim=[50],
    maxlen=[100],
)


def extract(df, dataset):
    ds_df = df[df["source"] == dataset]
    ds_sentences = ds_df["sentence"].values
    labs = ds_df["label"].values
    return (ds_df, ds_sentences, labs)


def get_train_and_test(sentences, y, test_sz=0.25, rs=1000):
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=test_sz, random_state=rs
    )
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)

    return (x_train, x_test, y_train, y_test)


def plot_history(history, source):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1)

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="Training acc")
    plt.plot(x, val_acc, "r", label="Validation acc")
    plt.title(f"Training and validation accuracy for {source}")
    plt.xlabel("Batch number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title(f"Training and validation loss for {source}")
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.legend()
    return plt


# initialize the entire dataframe from the data folder
df_list = []
for source, filepath in data_path.items():
    df = pd.read_csv(filepath, names=["sentence", "label"], sep="\t")
    df["source"] = source
    df_list.append(df)
df = pd.concat(df_list)


def baseline():
    print("_________Baseline results______________\n")
    for source in df["source"].unique():
        (_, sentences, y) = extract(df, source)
        (x_train, x_test, y_train, y_test) = get_train_and_test(sentences, y)

        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        accuracy = classifier.score(x_test, y_test)

        print(f"Accuracy for {source.capitalize()}: {accuracy*100}%")
    print("_______________________________________\n")


def first_keras():
    plots = []
    for source in df["source"].unique():
        (_, sentences, y) = extract(df, source)
        (x_train, x_test, y_train, y_test) = get_train_and_test(sentences, y)

        input_dim = x_train.shape[1]  # num features
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
        model.add(layers.Dense(1, activation="tanh"))

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=30,
            verbose=False,
            validation_data=(x_test, y_test),
            batch_size=20,
        )

        loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        plt = plot_history(history, source.capitalize())
        plots.append(plt)
        print("next")
    for plot in plots:
        plot.show()


def cnn():
    for source in df["source"].unique():
        (_, sentences, y) = extract(df, source)
        (x_train, x_test, y_train, y_test) = get_train_and_test(sentences, y)

        embedding_dim = 100
        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(128, 5, activation="relu"))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.summary()


if __name__ == "__main__":
    first_keras()
