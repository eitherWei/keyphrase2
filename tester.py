
import time
from methods import *

start = time.time()
# wrapper function that holds are init instance
dataClass = DataClass()
# class for reading our files
ex = ExtractContent(dataClass)
# all the setup  found ex
ex = ex.initialiseSet()


keyTerms = list(ex.df.keyPhrases)
length = []
all_terms = []
for term in keyTerms:
    term = term.split(",")
    for t in term:
        print(t)
        all_terms.append(t)
        t = t.split("+")
        print(t[0].split())
        print()
        length.append(len(t[0].split()))


df = pd.DataFrame({"length" : length})
print(df.length)
print(df.length.describe())
#
# print(all_terms)
# print(len(all_terms))
