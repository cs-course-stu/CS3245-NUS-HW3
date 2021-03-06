#!/usr/bin/python3
import re
import nltk
import sys
import getopt
from searcher import Searcher


# global variable
topK = 10
rate = 0.01
phrasal = False  # operate phrase query
pivoted = False  # operate normalize according to the length of doc
score = False    # print docid with its score

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")
    print("options:\n"
          "  -d  dictionary file path\n"
          "  -p  postings file path\n"
          "  -q  queries file path\n"
          "  -o  search results file path\n"
          "  -t  number of the results returned by each query\n"
          "  -r  penalty rate of the pivoted normalized document length\n"
          "  -x  enable phrasal query\n"
          "  -n  enable using pivoted normalized document length\n"
          "  -s  enable printing score\n")

def run_search(dict_file, postings_file, queries_file, results_file,
               topK, rate, phrasal, pivoted, score):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below
    searcher = Searcher(dict_file, postings_file, topK = topK, rate=rate, phrasal=phrasal, pivoted=pivoted, score=score)

    first_line = True
    with open(queries_file, 'r') as fin, \
         open(results_file, 'w') as fout:
        for line in fin:
            result, score = searcher.search(line)
            result = map(str, result)

            if first_line:
                result = ' '.join(result)
                first_line = False
            else:
                result = '\n' + ' '.join(result)

            fout.write(result)

            if score:
                score = '\n' + ' '.join(map(str, score))
                fout.write(score)

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:t:r:xns')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    elif o == '-t':
        topK = int(a)
    elif o == '-r':
        rate = float(a)
    elif o == '-x':  # operate phrase query
        phrasal = True
    elif o == '-n':  # operate normalize according to the length of doc
        pivoted = True
    elif o == '-s':
        score = True
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output,
           topK, rate,  phrasal, pivoted, score)
