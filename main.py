###### classes
# one class to store file
# one class that inherits store that processes files
import os
from methods import *
import time as time
import pandas as pd

from collections import Counter
from nltk.metrics.scores import accuracy, precision, recall, f_measure


###############################
###
# method looks at
# references
expandRefs = True
# acronyms
# deliminators : default
# stemming : default
# stopwords : N/A
################################

start = time.time()
# wrapper function that holds are init instance
dataClass = DataClass()
# class for reading our files
ex = ExtractContent(dataClass)
# all the setup  found ex
ex = ex.initialiseSet()

#identify where References separates the two texts
ex.df['refCutPoint'] = ex.df.raw_files.apply(ex.splitByReference)

if expandRefs:
    # lambda function takes cut off point and returns all else
    ex.df['ref_text'] = ex.df.apply(lambda row: row.raw_files[0][row.refCutPoint:], axis = 1)
    ex.df['pre_ref'] = ex.df.apply(lambda row: row.raw_files[0][:row.refCutPoint], axis = 1)
    # separates out references
    ex.df.pre_ref.apply(ex.extractReferencesTerms)
    # assigns a value to each reference term
    ex.processReferenceTerms()
    # create a dictionary of processed refs
    #print(ex.df.pre_ref[0])
    ex.df['processedRefs'] = ex.df.ref_text.apply(ex.cleanReferenceStrings)
    # fill out the text
    #ex.df['raw_text_ref'] = ex.df.pre_ref.apply(ex.expandColocatedRefs)
    ex.df['raw_text_ref'] = ex.df.raw_files.apply(ex.convertToString)
    ex.df['raw_text_ref'] = ex.fillOutReferences()
    #print(4*"\n")
    #print(10*"-")
    #print(ex.df.raw_text_ref[0])


# clean the data
try:
    ex.df['cleanText'] = ex.df.raw_text_ref.apply(ex.cleanTextMethod)
except:
    ex.df['raw_files'] = ex.df.raw_files.apply(ex.convertToString)
    ex.df['cleanText'] = ex.df.raw_files.apply(ex.cleanTextMethod)





# for ind  in range(0, len(ex.df.refCutPoint)):
#     cut = ex.df.refCutPoint[ind]
#     text = ex.df.cleanText[ind]
#     print(10*"-")
#     print(len(text))
#     art_text = text[:cut]
#     ref_text = text[cut:]
#     print(len(art_text))
#     print(len(ref_text))
#     print(10*"=")




# at this point we need to iterate over entire text

iter1 = 0
total = 0
total_recall = 0
total_precision = 0
current_total = 0

# for holding all phrases
all_phrases = []
all_cups = [0, 0, 0, 0, 0, 0]
for text in ex.df.cleanText[:1]:
    print("working on stage {}".format(iter1))
    # this creates TR for everything so expecting it to be time consuming
    #ex.df['scores'] = ex.df.cleanText.apply(ex.calculateDocumentScores)

    # iterates over text and performs textRank
    scores = ex.calculateDocumentScores(text)
    print("scores generated")

    # # generate the potential phrases
    #text = ex.df.cleanText[0]
    # # takes the text and generates all possible phrases - ex.candidatePhrases
    ex.generatePhrases(text)

    # take the phrases and weight them using scores from above - ex.weightedPhrases
    ex.weightPhrases(scores)

    rankedPhrases = dict(sorted(ex.weightedPhrases.items(), key=lambda x: x[1], reverse = True))
    y_pred = dict(list(rankedPhrases.items())[:15])
    y_true = ex.df.keyPhrases[iter1]

    y_true = ex.processKeyPhrases(y_true, y_pred)

    #print(y_true)
    #y_true = set(y_true)
    # extracts the index location for the current check
    cups = ex.determineIndexLocation(y_true, rankedPhrases)
    print(cups)
    # concatenate list for overall insight
    all_cups = [x + y for x, y in zip(all_cups, cups)]
    print(all_cups)


    current_total  = f_measure(y_true, set(y_pred.keys()))
    total_precision += precision(y_true, set(y_pred.keys()))
    total_recall += recall(y_true, set(y_pred.keys()))
    total += current_total

    print("total for round {}".format(str(current_total)))

    # increment iterator
    iter1 = iter1  + 1
    #get a record of all phrases
    all_phrases.extend(ex.candidatePhrases.keys())
    # reset stored variables
    ex.weightedPhrases = {}
    ex.candidatePhrases = {}



print("total fmeasure for model {}".format(total))
print("total recall for model {}".format(total_recall))
print("total precision for model {}".format(total_precision))
print(all_cups)

print("total_number of phrases for investigation {}".format(str(len(all_phrases))))
unique_phrases = len(list(set(all_phrases)))
print("total_number of unique phrases for investigation {}".format(str(len(unique_phrases))))

print(10*"-X-")
print((time.time() - start)/60)
#[189, 285, 218, 88, 321, 342] - 12.877704145496164 - standard
#[185, 273, 222, 85, 336, 342] - 12.643620761184584 - all_refs
# [183, 270, 214, 89, 337, 350] - 12.478250780825409 - mixed :/
# [189, 262, 215, 90, 337, 350] - 12.906800382358883 - stopwords included
# [189, 257, 214, 86, 358, 339] - length 4
# [195, 215, 176, 54, 464, 339] - length 4 , times the lenght of the phrase - 13.337510241405631
# alt textRank
# [196, 218, 170, 61, 459, 339] - 13.413850174078416
# altering the dash in finding phrase
# [205, 227, 178, 64, 511, 258] - 14.0019842973169
# without reference expan
# [218, 242, 163, 79, 491, 250] - 14.885092492515968
# formuala above was in correct .. we added in freq of phrase and adjusted - detenction but not yet the weighting
# [205, 246, 203, 76, 560, 153] - 14.009116019344257
# [240, 265, 157, 73, 532, 176]  - 16.297252509929884 - no references and no stopwords in phrases
# [232, 247, 170, 63, 548, 183] - 15.808600415260317 - stopwords and refs
# [233, 229, 160, 54, 584, 183] - with concat distinction
