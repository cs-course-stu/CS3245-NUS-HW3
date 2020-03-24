#!/usr/bin/python3
import re
import nltk
import sys
import getopt
from indexer import Indexer


# global variable
phrasal_query = False # operate phrase query
normalize = False # operate normalize according to the length of doc

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings, phrasal_query, normalize):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    # initialize the class
    inverted_index = Indexer(out_dict, out_postings, phrasal_query, normalize)
    inverted_index.build_index(in_dir)

    # save to file
    inverted_index.SavetoFile()

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:xn')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    elif o == '-x':  # operate phrase query
        phrasal_query = True
    elif o == '-n':  # operate normalize according to the length of doc
        normalize = True
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary,
            output_file_postings, phrasal_query, normalize)
