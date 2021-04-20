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
# 42 6f 6f 6c 65 61 6e 4d 6f 64 65 6c                                                                              #
#                                                                                                                     #
# Python 3.8.2                                                                                                        #
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

###################################################### CONSTANTS ######################################################

print("Initializing constants...")

INVERTED_INDEX_FILE_PATH = "indice.txt"
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
query = list()


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


def getQuery(queryFilePath):
    query = list()

    try:
        queryFileText = open(queryFilePath, "r").read()
        for term in queryFileText.split(OR_CHAR):
            subTerms = list()
            for subTerm in term.split(AND_CHAR):
                subTerms.append(subTerm.strip().lower())
            query.append(subTerms)

        return query
    except:
        print("Error processing query!")

        return None


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


def main():
    global words, data, invertedIndex, invertedIndexFileText, query

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

    query = getQuery(sys.argv[1:][1])

    if query is None:
        return

    print("Processing search...")

    matchedFileIds = set()
    for term in query:
        tempFileIds = set(data.keys())
        for subTerm in term:
            if subTerm.find(NEG_CHAR) == -1:
                foundFileIds = [index[0] for index in invertedIndex.get(STEMMER.stem(subTerm)) or []]
            else:
                foundFileIds = [fileId for fileId in list(data.keys()) if fileId not in [index[0] for index in invertedIndex.get(STEMMER.stem(subTerm[1:])) or []]]
            if len(foundFileIds) > 0:
                tempFileIds = tempFileIds.intersection(set(foundFileIds))
        matchedFileIds.update(tempFileIds)

    print("Writing result file...")
    finalFile = open(RESULT_FILE_PATH, "w")
    finalFile.write("{}\n{}".format(len(matchedFileIds), "\n".join([inputFilesPath[fileId - 1].replace("./", "") for fileId in matchedFileIds])))

    print("Done!")


######################################################## MAIN #########################################################

main()

#######################################################################################################################
