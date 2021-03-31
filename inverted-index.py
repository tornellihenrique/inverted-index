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
# 49 6E 76 65 72 74 65 64 49 6E 64 65 78                                                                              #
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

###################################################### CONSTANTS ######################################################

print("Initializing constants...")

FINAL_FILE_PATH = "indice.txt"
SLASH = "/"
NEW_LINE = "\n"
STOP_WORDS = corpus.stopwords.words('portuguese')
STEMMER = stem.RSLPStemmer()
UNIGRAM_TAGGER = tag.UnigramTagger(corpus.mac_morpho.tagged_sents())

################################################## GLOBAL VARIABLES ###################################################

basePath = str()
data = dict()
words = set()
invertedIndex = str()


######################################################## CODE #########################################################

def getInputFilesPath(baseFilePath):
    global basePath

    basePath = SLASH.join(baseFilePath.split(SLASH)[:-1])

    try:
        baseFile = open(baseFilePath, "r")
    except:
        return None
    else:
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
            word.isalpha() and word not in STOP_WORDS]


def handleWords(wordList):
    return [STEMMER.stem(word[0]) for word in UNIGRAM_TAGGER.tag(wordList) if word[1] not in ('PREP', 'ADV', 'ART')]


def getWordOccurrencesCount(word, wordList):
    count = 0

    for w in wordList:
        if w == word:
            count = count + 1

    return count


def main():
    global words, data, invertedIndex

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
                occurrences.append("{},{}".format(fileId, count))

        invertedIndex = invertedIndex + "{}: {}\n".format(word, " ".join(occurrences))

    print("Writing final file...")
    finalFile = open(FINAL_FILE_PATH, "w")
    finalFile.write(invertedIndex)

    print("Done!")


######################################################## MAIN #########################################################

main()

#######################################################################################################################
