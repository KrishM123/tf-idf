import nltk
import sys
import os
import string
import math
import copy

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    files = os.listdir(os.getcwd() + os.sep + directory + os.sep)
    dictionary = {}
    for file in files:
        dictionary[file] = open(os.getcwd() + os.sep + directory + os.sep + file, encoding='UTF-8').read()
    return dictionary


def tokenize(document):
    words = nltk.tokenize.word_tokenize(document.lower())
    punctuation = list(string.punctuation)
    stop_words = nltk.corpus.stopwords.words("english")
    pop = []
    for pos in range(0, len(words)-1):
        if words[pos] in punctuation or words[pos] in stop_words:
            pop.append(pos)
    updated_words = []
    for pos in range(0, len(words)-1):
        if pos not in pop:
            updated_words.append(words[pos])
    return updated_words


def compute_idfs(documents):
    values = dict()

    for file in documents:
        words = set()

        for word in documents[file]:
            if word not in words:
                words.add(word)
                try:
                    values[word] += 1
                except KeyError:
                    values[word] = 1

    return {word: math.log(len(documents) / values[word]) for word in values}


def top_files(query, files, idfs, n):
    tf_idfs = dict()

    for file in files:
        tf_idfs[file] = 0
        for word in query:
            tf_idfs[file] += files[file].count(word) * idfs[word]

    final = [key for key, value in sorted(tf_idfs.items(), key=lambda item: item[1], reverse=True)][:n]
    return final


def top_sentences(query, sentences, idfs, n):
    ranks = list()

    for sentence in sentences:
        sentence_values = [sentence, 0, 0]

        for word in query:
            if word in sentences[sentence]:
                sentence_values[1] += idfs[word]
                sentence_values[2] += sentences[sentence].count(word) / len(sentences[sentence])

        ranks.append(sentence_values)

    final = [sentence for sentence, mwm, qtd in sorted(ranks, key=lambda item: (item[1], item[2]), reverse=True)][:n]
    return final


if __name__ == "__main__":
    main()
