# method reads content from file

import os
import codecs
import pandas as pd
import string
import re
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import nltk
import networkx as nx
from collections import Counter
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))


class DataClass():
    ''' class that holds our attributes '''
    ''' df -> main datasource '''
    ''' methods setDataFrame '''

    def __init__(self, path= None, df = None):
        self.df = pd.DataFrame()
        self.name = "tester name"
        self.path = path
        # keeps a store of gold standard set
        self.keyPhrases = {}
        # contains freq of phrase
        self.candidatePhrases = {}
        #  calculatePhraseValues
        self.weightedPhrases = {}
        # a dictionary to hold all text terms for ref expansion
        self.all_terms = {}
        # a dictionary to hold term values
        self.book = {}

    def setDataFrame(self, files, fileName):
        self.df[fileName] = files


class ExtractContent(DataClass):
    ''' child class fo dataClass '''
    ''' reads files '''

    def allocatePhrases(self, text):
        return self.keyPhrases[text]

    def stripTags(self, text):
        text = text.split(".")
        return text[0]

    def readFileNames(self, path):
        dirs = os.listdir(path)
        return dirs

    def createFileTag(self, text):
        text = self.path + text
        return text

    def extractContent(self, file_path):
        # given the file name it pulls out the content
        doc_array = []
        with codecs.open(file_path, 'r', encoding='utf8', errors = 'ignore') as file:
            line = file.read()
            doc_array.append(line)

        return doc_array

    def splitByReference(self, text):
        "REFERENCES"
        pattern = "REFERENCES"
        counter  = 0
        #x = re.findall(pattern, text[0])
        y = text[0].find(pattern)
        # print(text[0][y:])
        # print(2*"\n")
        #print(string.punctuation)
        return(y)

    def divideByDeliminator(self, text):
        # replace sentence delims with breakers

        text = text.replace(".", " _ ")
        text = text.replace(",", " _ ")
        text = text.replace(":", " _ ")
        text = text.replace(";", " _ ")
        return text

    def returnTermWeight(self, term , scores):
        weight = 0
        if "-" in term:
            tempTerm = term.split("-")
            for term in tempTerm:
                if term in scores:
                    weight += scores[term]
        elif term in scores:
            weight += scores[term]

        return weight

    def extractAccs(self, text):
        accDict = {}
        # acc pattern
        pattern = pattern = '\((.*?)\)'
        text = text[0].split()
        #print(text)
        # iterate over the array
        for i in range(0, len(text)):
            cand = re.search(pattern, text[i])
            if cand:
                candidate = cand.group(1)
                if len(candidate) > 1:
                    # check previous in the array if acc defined
                    j = len(candidate)
                    cand_terms = text[i-j:i]
                    headerString = self.extractFirstLetters(cand_terms)
                    if candidate.lower() == headerString.lower():
                        accDict[candidate.lower()] = cand_terms
        return accDict


    def fillOutAcc(self, text, dict):
        new_text = [""]
        for term in text.split():
            removeSym = string.punctuation
            # removes pattern and lowercases
            pattern = r"[{}]".format(removeSym)
            term1 = re.sub(pattern, " ", term.lower())
            new_text.append(term)
            if term1 in dict:
                i = 1
                new_text.append(dict[term1])
        #print(new_text)
        new_text =  " ".join(map(str, new_text[:100]))




    def extractFirstLetters(self, phrase):
        headerString  = ""
        for term in phrase:
            headerString += term[:1]
        return headerString


    def weightPhrases(self, scores):
        # candidatePhrases contains freq and phrase
        for phrase , freq in self.candidatePhrases.items():
            weight = 0
            # iterate over phrase and check if term has been weighted
            # using pageRank
            for term in phrase.split():
                if term in scores:
                    # pass to another method to perform
                    weight += self.returnTermWeight(term, scores)
            weight = weight/len(phrase.split())
            self.weightedPhrases[phrase] = weight * (len(phrase.split()) * self.candidatePhrases[phrase])


    def determineIndexLocation(self, y_pred, weightedPhrases):
        # get the ordering of the terms in the dict
        index = [x for x in range(1 , len(weightedPhrases.items())+ 1)]
        # assign each term a location in the dict
        docDict = dict(zip(weightedPhrases.keys(), index))
        # create a dictionary that holds location
        tempDict = {}
        for value in y_pred:
            if value in docDict:
                tempDict[value] = docDict[value]
            else:
                tempDict[value] = - 1

        print(tempDict)
        cups = self.rankLocationIndex([tempDict.values()])
        return cups

    def rankLocationIndex(self, indexList):
        cups = [15, 100, 500, 1000, 0]
        notPresent = 0
        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        for docIndex in indexList:
            for index in docIndex:
                if index == -1:
                    notPresent = notPresent + 1
                elif index <= cups[0]:
                    zero = zero + 1
                elif index <= cups[1]:
                    one = one + 1
                elif index <= cups[2]:
                    two = two + 1
                elif index <= cups[3]:
                    three = three + 1
                else:
                    four = four + 1

        return [zero, one, two, three, four, notPresent]




    def cleanText(self, text):
        # remove "\n"
        text = " ".join(text.split("\n"))
        # initialise punctuation
        removeSym = string.punctuation
        # remove desired case
        removeSym = removeSym.replace("_", "")
        removeSym = removeSym.replace("-", "")
        # removes pattern and lowercases
        pattern = r"[{}]".format(removeSym)

        text = re.sub(pattern, " ", text.lower())

        # removes singletons
        text = " ".join([x for x in text.split() if len(x) > 1 or x == "_"])

        # remove all integers not connected to a constonant
        pattern = (r"\d")
        #text = re.sub(pattern, " ", text)
        # takes out all integers not connected with a consonant
        text = [x for x in text.split() if not x.isdigit()]

        return text

    def posDoc(self, text):
        text =  nltk.pos_tag(text)

        return text

    def alternativePos(self, text):
        taggedText = []
        for term in text:
            tag = nltk.pos_tag(term)
            taggedText.append(tag[0])

        return taggedText

    def formatArrayforGraph(self, friendlyPOS, text):
        array = []
        for term in text:
            if term in friendlyPOS or term == "_":
                # append a stemmed instance
                array.append(ps.stem(term))
            else:
                array.append("_")
        return array

    def plotToGraph(self, array):
        g = nx.Graph()
        # join and break array to emphasis "_"
        array = " ".join(array)
        # splits array but creates \s and empty arrays
        array = array.split("_")
        array = [x.strip() for x in array]
        for sect in array:
            temp = sect.split()
            while len(temp) > 1:
                g , temp = self.plotByArray(temp, g)

        return g

    def calculatePageRank(self, graph):
        initial_value = 1
        scores = dict.fromkeys(graph.nodes(), initial_value)
        print("there are {} nodes".format(len(graph.nodes())))


        iter_value = 0
        converge = 0
        for index in range(100):
            # count number of loops
            iter_value += 1
            for i in graph.nodes():
                rank = 1 - .85
                for j in graph.neighbors(i):
                    neighbors_sum = sum(graph.edge[j][k]['weight']  for k in graph.neighbors(j))
                    rank += scores[j] * graph.edge[j][i]['weight']/ neighbors_sum

                if abs(scores[i] - rank) <= 0.0001:
                    converge += 1

                scores[i] = .85 *  rank

                iter_value += 1

            if converge == len(graph.nodes()):
                break

        return scores

    def handleDashStem(self, sent):
        # this method loops over text if it sees a "-"
        # it processes accordingly
        new_sent = []
        for word in sent:
            if "-" in word:
                word = word.split("-")
                word = [ps.stem(w) for w in word]
                word = "-".join(word)
                new_sent.append(word)
            else:
                new_sent.append(ps.stem(word))

        return new_sent




    def generatePhrases(self, text):
        # separates out the text and breaks into string sentences
        text = " ".join(text)
        text = text.split("_")
        # removes the whitespace and null strings
        text = [x.strip() for x in text if len(x) > 1]
        # iterates over each sent
        for sent in text:
            # split sent and do while there is still a word
            # recursive method so string gets one word short per run
            sent = sent.split()
            sent = [x for x in sent if x not in stop]
            # stem the sentences here
            # check for dashes
            sent = self.handleDashStem(sent)

            #sent = [ps.stem(x) for x in sent]
            while len(sent) > 0:
                sent = self.generateSentPhrases(sent)

        #self.candidatePhrases =
        return self.candidatePhrases

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    def analysisRefText(self, text, answers):

        print(text)
        values = []

        for value in text.values():
            values.append(value)
        print(values)
        for v in values:
            v = v.split()


        # clean all terms
        # text = self.cleanText(text)
        # termsDict = dict(Counter(text))
        # print(termsDict)
        # docDict = dict(sorted(termsDict.items(), key=lambda x: x[1], reverse = True))
        # print(docDict)
        # print(answers)




    def generateSentPhrases(self,sent):
        # take a chunk length the size of desired phrase
        text = sent[:5]
        # works down to a single word
        index = 1
        while index < len(text)+1:
            # index starts at 1 and works to length
            # ensuring each term gets counted
            phrase = text[:index]
            index = index + 1
            phrase = " ".join(phrase)
            # check the global dictionary and act appropriately
            if phrase in self.candidatePhrases:
                self.candidatePhrases[phrase] += 1
            else:
                self.candidatePhrases[phrase] = 1

        return sent[1:]


    def convertToString(self,text):
        return " ".join(text)




    def cleanTextMethod(self, text):
        # implement deliminator text
        text = self.divideByDeliminator(text)
        # processing the text
        text = self.cleanText(text)

        return text


    def calculateDocumentScores(self, text):

        # apply the stemming process
        posFriendly = self.applyPoS(text)
        # print(text)
        # # create a graph for TextRank
        graphArray = self.formatArrayforGraph(posFriendly, text)
        graph = self.plotToGraph(graphArray)
        scores = self.calculatePageRank(graph)
        scores = dict(sorted(scores.items(), key = lambda x : x[1], reverse = True))

        return scores

    def makeSignal(self):

        beep = lambda x: os.system("echo -n '\a';sleep 0.2;" * x)
        beep(3)


    def plotByArray(self, array, g):
        target = array[0]
        processArray = array[0:4]
        #g.add_node(target)
        for i in range(1,len(processArray)):
            try:
                g[target][array[i]]['weight'] += 1/i
            except:
                g.add_edge(target, array[i], weight = 1/i)
        return( g, array[1:])


    def applyPoS(self, text):

        text = self.posDoc(text)
        text1 = self.alternativePos(text)

        friendlyList = ["NN", "JJ", "NNS", "NNP", "JJS", "NNPS", "JJR"]

        keepList = [x[0] for x in text if x[1] in friendlyList]
        keepList1 = [x[0] for x in text1 if x[1] in friendlyList]

        keeptotal = keepList + keepList1

        keepListT = list(set(keeptotal))

        return keepListT



    def initialiseSet(self):

        # wrapper function that holds are init instance
        dataClass = DataClass()
        # class for reading our files
        ex = ExtractContent(dataClass)
        # initialise location of path
        path = "/Users/stephenbradshaw/Documents/codingTest/semevalTests/SemEval2010/test/"
        answer_path = "/Users/stephenbradshaw/Documents/codingTest/semevalTests/SemEval2010/test_answer/"
        ex.path = path
        # read in the files
        files = ex.readFileNames(path)
        # # assign the files to an array in df
        ex.setDataFrame(files, "fileNames")
        # # create a full index of the files
        filePaths = ex.df.fileNames.apply(ex.createFileTag)
        # # create a column for file paths
        ex.setDataFrame(filePaths, "filePaths")
        # # extract the full content
        raw_files = ex.df.filePaths.apply(ex.extractContent)
        # set the files to the dataframe
        ex.setDataFrame(raw_files, "raw_files")
        # extract the tags
        tags = ex.df.fileNames.apply(ex.stripTags)
        # assign tags
        ex.setDataFrame(tags, "tags")
        # extract keyPhrases
        #####
        # read the answerFiles
        combinedFile = list(ex.readFileNames(answer_path))[2]
        combinedFile = answer_path + combinedFile
        answers = ex.extractContent(combinedFile)
        answers = answers[0].split("\n")

        keys = []
        terms = []
        for answer in answers:
            answer = answer.split(":")
            if len(answer) > 1:
                keys.append(answer[0].strip())
                terms.append(answer[1].strip())

        ###
        ex.keyPhrases = dict(zip(keys, terms))

        ex.df['keyPhrases'] = ex.df.tags.apply(ex.allocatePhrases)

        return ex


    def processReferenceTerms(self):
        # calculates the value of each term
        total = sum(self.all_terms.values())
        for term , value in self.all_terms.items():
            self.book[term] = value/total

    def returnProb(self, term):
        if term in self.book:
            return self.book[term]
        else:
            return 0


    def cleanReferenceStrings(self, text):
        pattern = '(\[.*?\])'
        text = re.split(pattern, text)
        citeDict = {}
        for i in range(0, len(text)):
            if i % 2 == 1:
                textStr = self.cleanText(text[i + 1])
                text1 = [x for x in textStr if self.returnProb(x) > 0.0001]
                citeDict[text[i]] = " ".join(text1)

        return citeDict

    def fillOutReferences(self):
        all_text = []
        # iterates over dict rows
        for index, row in self.df.iterrows():
            # extract the document
            text = self.df.raw_text_ref[index]
            # extract the relevant dictionary
            ref_dict = self.df.processedRefs[index]
            # loop over the dictionary and update each key
            for key, value in ref_dict.items():
                value = " _ " + value + " _ "
                text = text.replace(key, value)
            all_text.append(text)

        return all_text

    def expandColocatedRefs(self, text):
        pattern = pattern = '(\[.*?\]+)'
        result = re.findall(pattern, text)
        removeSym = string.punctuation
        pattern2 = r"[{}]".format(removeSym)
        for value1 in result:
            value = re.sub(pattern2, "", "".join(value1))
            value = value.strip()
            value = value.split(" ")
            boolean = False
            new_format = []
            for v in value:
                if v.isdigit():
                    boolean = True
                else:
                    break
            if boolean:
                for v in value:
                    new_format.append("[" + v + "]")
                text = text.replace(value1 , " ".join(new_format))
        return text










    def extractReferencesTerms(self, text):
        text = self.cleanText(text)

        textDict = dict(Counter(text))
        # # extract all terms and update dictionary
        for term, value in textDict.items():
            if term in self.all_terms:
                self.all_terms[term] += value
            else:
                self.all_terms[term] = 1



    def processKeyPhrases(self, text, y_pred):
        text = text.split(",")
        array = []
        for t in text:
            if "+" in t:
                altOption = t.split("+")
                if altOption[0] in y_pred:
                    t = altOption[0]
                else:
                    t = altOption[1]
            array.append(t)


        return set(array)
