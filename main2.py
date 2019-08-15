from methods_main4 import *
from methods import *

import pandas as pd

from collections import Counter
from nltk.metrics.scores import accuracy, precision, recall, f_measure

import os

import time

start = time.time()
expandRefs = True
expandAcc = True
# wrapper function that holds are init instance
dataClass = DataClass()
# class for reading our files
ex = ExtractContent(dataClass)
# all the setup  found ex
ex = ex.initialiseSet()


total = 0
total_recall = 0
total_precision = 0
current_total = 0


if expandAcc:
    files_list = []
    for i in range(len(ex.df.raw_files)):
        text = ex.df.raw_files[i]
        accDict = ex.extractAccs(text)
        accFile = ex.fillOutAcc( text[0], accDict)
        files_list.append(accFile)

    ex.df.raw_files = files_list


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
    #ßex.analysisRefText(ex.df['processedRefs'][0], ex.df.keyPhrases[0])


    # fill out the text
    ex.df['raw_text_ref'] = ex.df.pre_ref.apply(ex.expandColocatedRefs)
    #ex.df['raw_text_ref'] = ex.df.raw_files.apply(ex.convertToString)
    ex.df['raw_text_ref'] = ex.fillOutReferences()
    #print(4*"\n")
    #print(10*"-")
    #print(ex.df.raw_text_ref[0])

# create and instance of dataClass
path = ""
dataClass = DataSet(path)


# clean the data
try:
    ex.df['cleanText'] = ex.df.raw_text_ref.apply(ex.cleanTextMethod)
except:
    ex.df['raw_files'] = ex.df.raw_files.apply(ex.convertToString)
    ex.df['cleanText'] = ex.df.raw_files.apply(ex.cleanTextMethod)

iter1 = 0
all_cups = [0, 0, 0, 0, 0, 0]
for text in ex.df.cleanText:
    print("on stage {}".format(iter1))

    PR = pageRankClass(text)
    #print(text)
    PR.constructGraph(text)
    print("number of nodes : " + str(len(PR.graph.nodes())))

    # # takes the text and generates all possible phrases - ex.candidatePhrases
    ex.generatePhrases(text)

    print("here ? ")
    scores = PR.textRankDict

    ex.weightPhrases(scores)

    rankedPhrases = dict(sorted(ex.weightedPhrases.items(), key=lambda x: x[1], reverse = True))
    y_pred = dict(list(rankedPhrases.items())[:15])
    y_true = ex.df.keyPhrases[iter1]

    y_true = ex.processKeyPhrases(y_true, y_pred)

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
    #all_phrases.extend(ex.candidatePhrases.keys())
    # reset stored variables
    ex.weightedPhrases = {}
    ex.candidatePhrases = {}

    print(y_pred)



print("total fmeasure for model {}".format(total))
print("total recall for model {}".format(total_recall))
print("total precision for model {}".format(total_precision))
print(all_cups)




print(10*"-x-")
print((time.time() - start)/60)
