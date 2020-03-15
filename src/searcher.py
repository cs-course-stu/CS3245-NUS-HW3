#!/usr/bin/python3

import sys
import math
import nltk
import numpy as np
# from indexer import Indexer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class Searcher:
    """ Searcher is a class dealing with real-time querying.
        It implements the ranked retrieval based on the VSM(Vector Space Model).
        It also support Phrasal Queries and Pivoted Normalized Document Length.

    Args:
        dictionary_file: the file path of the dictionary
        postings_file: the file path of the postings
        phrasal: boolean indicator for dealing queries as phrasal queries
        pivoted: boolean indicator for using pivoted normalized document length
    """

    def __init__(self, dictionary_file, postings_file, phrasal = False, pivoted = False):
        self.dictionary_file = dictionary_file
        self.postings_file = postings_file
        self.phrasal = phrasal
        self.pivoted = pivoted

        self.stemmer = PorterStemmer()
        # self.indexer = Indexer(dictionary_file, postings_file, phrasal, pivoted)

    """ Search and return docIds according to the boolean expression.

    Args:
        query: the query string

    Returns:
        result: the list of 10 most relevant docIds in response to the query
    """
    def search(self, query):
        # step 1: tokenize the query to get all the terms
        terms, tokens = self._tokenize(query)

        # step 2: get the postings lists of the terms
        postings_lists = self.indexer(terms)

        # step 3: get the docs that need to rank
        if not phrasal:
            # text query
            # Step 3-1: get total relevant docIds
            condidate = self._universe(terms, postings_lists)
        else:
            # phrasal query
            # step 3-1: get all the docs that contains all the terms in the query
            condidate = self._intersection(terms, postings_lists)

            # step 3-2: judging every doc whether it contains the phrase
            pass

        # step 4: pass the condidate docs to the rank function get the result
        retult = self.rank(query, candidate, postings_lists)

        # step 5: return the result
        return result

    """ Rank the documents and return the 10 most relevant docIds.
        The result should be in the order of relevant.

    Args:
       terms: all terms in the query string
       doc_list: the list of candidate docs
       postings_lists: the dictionary with terms to posting lists mapping

    Returns:
       result: the list of 10 most relevant docIds in response to the query
    """
    def rank(self, query, doc_list, postings_lists):
        # step 1: initialze the max-heap

        # step 2: construct and normalize the query vector

        # step 3: construct document vector based on the weights

        # step 4: use Scorer to evaluate the document

        # step 5: add the document to the heap

        # step 6: return doc
        pass

    """ Get the universe of the docs that appear in one of the lists.

    Args:
        terms: all terms in the query string
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        universe: the universe of the docs that appear in one of the lists
    """
    def _get_universe(self, terms, postings_lists):
        universe = set()
        for term in terms:
            postings = postings_lists[term][0]
            length = postings.shape[0]
            for i in range(0, length):
                universe.add(postings[i][0])

        universe = list(universe)
        return universe

    """ Get the intersection of docs

    Args:
        terms: all terms in the query string
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        intersection: the intersection of postings lists
    """
    def _get_intersection(self):
        pass

    """ Merge the two sets passed in based on the op type.

    Args:
        op: the type of merging operation
        set1: the set on the left hand side to be merged
        set2: the set on the right hand side to be merged

    Returns:
        resutlt: the merged result
    """
    def _merge(self, ):
        pass

    """ Tokenize the query into single term.

    Args:
        query: the query string

    Returns:
        terms: a set contains all the terms appeared in the query string
        tokens: a list contains all the tokens appeared in the query string
    """
    def _tokenize(self, query):
        # tokenize the query string
        tokens = [word for sent in nltk.sent_tokenize(query)
                       for word in nltk.word_tokenize(sent)]

        # stem the tokens
        tokens = [self.stemmer.stem(token).lower() for token in tokens]

        # get the set of tokens
        terms = set(tokens)
        terms = list(terms)

        return terms, tokens

if __name__ == '__main__':
    # Create a Searcher
    searcher = Searcher('dictionary.txt', 'postings.txt', phrasal = False, pivoted = False)

    terms = ['into', 'queri', 'can', 'term', 'searcher', 'token', 'string', 'and']
    postings_lists = {
        'into'    : (np.array([[1, 5], [3, 6], [5, 1]]), ),
        'queri'   : (np.array([[0, 5]]), ),
        'can'     : (np.array([[7, 10], [9, 3]]), ),
        'term'    : (np.array([[2, 5], [4, 6], [6, 10]]), ),
        'searcher': (np.array([[8, 3]]), ),
        'token'   : (np.array([[1, 7], [4, 6], [7, 3]]), ),
        'string'  : (np.array([[2, 5], [5, 6], [8, 1]]), ),
        'and'     : (np.array([[3, 3], [6, 6], [9, 9]]), )
    }

    test = '_get_universe'
    # Test tokenizing the query string
    if test == '_tokenize':
        terms, tokens = searcher._tokenize('Searcher can tokenize query strings into terms and tokens')
        print(terms)
        print(tokens)
    elif test == '_get_universe':
        universe = searcher._get_universe(terms, postings_lists)
        print(universe)
