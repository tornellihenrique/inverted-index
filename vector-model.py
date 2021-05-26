###################################################### MADE BY ########################################################
#  _    _                 _                    _______                    _ _ _    _____                   _          #
# | |  | |               (_)                  |__   __|                  | | (_)  |  __ \                 | |         #
# | |__| | ___ _ __  _ __ _  __ _ _   _  ___     | | ___  _ __ _ __   ___| | |_   | |  | |_   _  __ _ _ __| |_ ___    #
# |  __  |/ _ \ '_ \| '__| |/ _` | | | |/ _ \    | |/ _ \| '__| '_ \ / _ \ | | |  | |  | | | | |/ _` | '__| __/ _ \   #
# | |  | |  __/ | | | |  | | (_| | |_| |  __/    | | (_) | |  | | | |  __/ | | |  | |__| | |_| | (_| | |  | ||  __/   #
# |_|  |_|\___|_| |_|_|  |_|\__, |\__,_|\___|    |_|\___/|_|  |_| |_|\___|_|_|_|  |_____/ \__,_|\__,_|_|   \__\___|   #
#                              | |                                                                                    #
#                              |_|                                                                                    #
# Registration:                                                                                                       #
# 31 31 38 31 31 42 53 49 32 30 32                                                                                    #
#                                                                                                                     #
# Project:                                                                                                            #
# 56 65 63 74 6f 72 4d 6f 64 65 6c                                                                                    #
#                                                                                                                     #
# Python 3.9.4                                                                                                        #
# [Clang 12.0.0 (clang-1200.0.32.29)] on darwin                                                                       #
#                                                                                                                     #
# “If you can't explain it to a six year old, you don't understand it yourself.”                                      #
# ― Albert Einstein                                                                                                   #
#                                                                                                                     #
# Let's begin...                                                                                                      #
#######################################################################################################################

####################################################### IMPORTS #######################################################

import sys
from nltk import tokenize
from nltk import corpus
from nltk import stem
from nltk import tag
import pickle
from math import log, sqrt

###################################################### CONSTANTS ######################################################

print("Initializing constants...")

INVERTED_INDEX_FILE_PATH = "indice.txt"
WEIGHTS_FILE_PATH = "pesos.txt"
RESULT_FILE_PATH = "resposta.txt"
SLASH = "/"
NEW_LINE = "\n"
PUNCTUATION = [".", "..", "...", ",", "!", "?", " "]
STOP_WORDS = corpus.stopwords.words('portuguese')
STEMMER = stem.RSLPStemmer()
UNIGRAM_TAGGER = None
UNIGRAM_TAGGER_BIN = "tagger.bin"
OR_CHAR = "|"
AND_CHAR = "&"
NEG_CHAR = "!"
LOG_BASE = 10
MIN_THRESHOLD = 1 / 1000

try:
    UNIGRAM_TAGGER = pickle.load(open(UNIGRAM_TAGGER_BIN, "rb"))
except:
    print("This can take a few moments...")
    UNIGRAM_TAGGER = tag.UnigramTagger(corpus.mac_morpho.tagged_sents())
    pickle.dump(UNIGRAM_TAGGER, open(UNIGRAM_TAGGER_BIN, "wb"))

################################################## GLOBAL VARIABLES ###################################################

basePath = str()
data = dict()
words = set()
invertedIndex = dict()
invertedIndexFileText = str()
booleanModel = dict()
idf = dict()
tfIdf = dict()
query = list()
similarity = list()


######################################################## CODE #########################################################

def getInputFilesPath(baseFilePath):
    global basePath

    basePath = SLASH.join(baseFilePath.split(SLASH)[:-1])

    try:
        baseFile = open(baseFilePath, "r")
    except:
        return None
    else:
        basePath = "." if basePath == "" else basePath
        return ["{}{}{}".format(basePath, SLASH, inputFile.replace(NEW_LINE, "")) for inputFile in baseFile.readlines()
                if
                inputFile != NEW_LINE]


def getFilePathWords(inputFilePath):
    try:
        inputFile = open(inputFilePath, "r")
    except:
        print("Error on reading {}!".format(inputFilePath))
        return []
    return [word.lower() for word in tokenize.word_tokenize(inputFile.read(), language='portuguese') if
            word not in PUNCTUATION and word not in STOP_WORDS]


def handleWords(wordList):
    taggedWords = UNIGRAM_TAGGER.tag(wordList)
    return [STEMMER.stem(word[0]) for word in taggedWords if word[1] not in ('PREP', 'KC', 'KS', 'ART')]


def getWordOccurrencesCount(word, wordList):
    count = 0

    for w in wordList:
        if w == word:
            count = count + 1

    return count


def calculateIDF(total, n):
    if n == 0:
        return 0

    return log(total / n, LOG_BASE)


def calculateTFIDF(freq, word):
    global idf

    if freq == 0:
        return 0

    return (1 + log(freq, LOG_BASE)) * idf.get(word)


def getWordFreqInFile(word, fileId):
    global invertedIndex

    aux = dict(invertedIndex.get(word))

    return aux.get(fileId) if aux.get(fileId) else 0


def getQuery(queryFilePath):
    query = list()

    try:
        queryFileText = open(queryFilePath, "r").read()
        for term in queryFileText.split(OR_CHAR):
            subTerms = list()
            for subTerm in term.split(AND_CHAR):
                subTerms.append(NEG_CHAR + STEMMER.stem(subTerm.strip().lower()[1:]) if subTerm.find(NEG_CHAR) != -1 else STEMMER.stem(subTerm.strip().lower()))
            query.append(subTerms)

        return query
    except:
        print("Error processing query!")

        return None


def calculateInternalProduct(weightsDict, queryDict):
    internalProduct = 0
    aux = 0
    for word in queryDict.keys():
        if word in weightsDict.keys():
            aux = weightsDict.get(word)
        else:
            aux = 0
        internalProduct += queryDict.get(word) * aux

    return internalProduct


def calculateEuclidianNorm(doc):
    return sqrt(sum([pow(doc[word], 2) for word in doc.keys()]))


def calculateSimilarity(weights, queryDict):
    weightsDict = dict(weights)

    # Calculating Internal Product
    internalProduct = calculateInternalProduct(weightsDict, queryDict)

    # Calculating Euclidian Normal
    euclidianCalc = calculateEuclidianNorm(weightsDict) * calculateEuclidianNorm(queryDict)

    return 0 if euclidianCalc == 0 else internalProduct / euclidianCalc


def main():
    global words, data, invertedIndex, invertedIndexFileText, query, booleanModel, idf, tfIdf, similarity

    inputFilesPath = getInputFilesPath(sys.argv[1:][0])

    if inputFilesPath is None:
        print("Error on reading base file! Exiting...")
        return

    print("Processing data...")
    fileId = 1
    for inputFilePath in inputFilesPath:
        currentWords = handleWords(getFilePathWords(inputFilePath))
        data.update({
            fileId: {
                "words": currentWords
            }
        })

        words.update(currentWords)

        fileId = fileId + 1

    for word in sorted(words):
        occurrences = list()

        for fileId in data.keys():
            count = getWordOccurrencesCount(word, data.get(fileId)["words"])

            if count > 0:
                occurrences.append((fileId, count))

        invertedIndex.update({
            word: occurrences
        })

        invertedIndexFileText = invertedIndexFileText + "{}: {}\n".format(word, " ".join(
            [str(occ[0]) + "," + str(occ[1]) for occ in occurrences]))

    print("Writing inverted index file...")
    finalFile = open(INVERTED_INDEX_FILE_PATH, "w")
    finalFile.write(invertedIndexFileText)

    print("Inverted Index processed!")

    print("Calculating IDF...")

    fileIds = data.keys()

    for word in words:
        idf.update({
            word: calculateIDF(len(fileIds), len(invertedIndex.get(word)))
        })

    print("Calculating TF-IDF...")

    for fileId in fileIds:
        tfIdf.update({
            fileId: [(word, calculateTFIDF(getWordFreqInFile(word, fileId), word)) for word in words]
        })

    print("Writing weights file...")
    finalFile = open(WEIGHTS_FILE_PATH, "w")
    finalFile.write("\n".join(["{}:\t{}".format(inputFilesPath[fileId - 1].replace("./", ""), "\t".join(["{}, {}".format(item[0], item[1]) for item in tfIdf.get(fileId) if item[1] > 0])) for fileId in fileIds]))

    query = getQuery(sys.argv[1:][1])

    if query is None:
        return

    if len(query) == 0:
        return

    print("Processing search...")

    queryDict = dict()
    for word in query[0]:
        queryDict.update({
            word: idf.get(word) or 0
        })

    for fileId in fileIds:
        res = calculateSimilarity(tfIdf.get(fileId), queryDict)
        if res > MIN_THRESHOLD:
            similarity.append((fileId, res))

    similarity.sort(key=lambda x: (-x[1], -x[1]))

    print("Writing result file...")
    finalFile = open(RESULT_FILE_PATH, "w")
    finalFile.write("{}\n{}".format(len(similarity), "\n".join(["{}\t{}".format(inputFilesPath[int(s[0]) - 1].replace("./", ""), s[1]) for s in similarity])))

    print("Done!")


######################################################## MAIN #########################################################

main()

#######################################################################################################################
