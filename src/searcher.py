#!/usr/bin/python3

import sys
import math
import nltk
import array
import numpy as np
from indexer import Indexer
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

class Searcher:
    """ Searcher is a class dealing with real-time querying.
        It implements the ranked retrieval based on the VSM(Vector Space Model).
        It also support Phrasal Queries and Pivoted Normalized Document Length.

    Args:
        dictionary_file: the file path of the dictionary
        postings_file: the file path of the postings
        topK: the number of highest-scoring documents to be returned
        phrasal: boolean indicator for dealing queries as phrasal queries
        pivoted: boolean indicator for using pivoted normalized document length
    """

    def __init__(self, dictionary_file, postings_file,
                 topK = 10, phrasal = False, pivoted = False):
        self.dictionary_file = dictionary_file
        self.postings_file = postings_file
        self.topK = topK
        self.phrasal = phrasal
        self.pivoted = pivoted

        self.stemmer = PorterStemmer()
        self.indexer = Indexer(dictionary_file, postings_file, phrasal, pivoted)
        self.average, self.total_doc, self.dictionary = self.indexer.LoadDict()
        # self.scorer = Scorer(self.average, pivoted)

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
        postings_lists = self.indexer.LoadTerms(terms)

        # step 3: get the docs that need to rank
        if not phrasal:
            # text query
            # Step 3-1: get total relevant docIds
            condidate = self._get_universe(terms, postings_lists)

            # step 3-2: judging every doc whether it contains the phrase
            pass

        else:
            # phrasal query
            # step 3-1: get all the docs that contains all the terms in the query
            condidate = self._get_intersection(terms, postings_lists)

        # step 4: construct the query vector
        query_vector = self._get_query_vector(terms, postings_lists)

        # step 5: pass the condidate docs to the rank function get the result
        retult = self.rank(query_vector, candidate, postings_lists)

        # step 6: return the result
        return result

    """ Rank the documents and return the 10 most relevant docIds.
        The result should be in the order of relevant.

    Args:
       query_vector: the constructed query vector
       doc_list: the list of candidate docs
       postings_lists: the dictionary with terms to posting lists mapping

    Returns:
       result: the list of 10 most relevant docIds in response to the query
    """
    def rank(self, query_vector, doc_list, postings_lists):
        # step 1: initialze the max-heap
        heap = []

        # step 2: processing every document
        for doc in doc_list:
            # step 2-1: construct document vector based on the weights
            doc_vector = self._get_doc_vector(doc, postings_lists)

            # step 2-2: use Scoreer to evaluate the document
            # score = self.scorer.evaluate(query_vector, doc_vector)
            score = 0

            # step 2-3: add the document to the heap
            if len(heap) < self.topK:
                heapq.heappush(heap, (score, doc))
            else:
                heapq.heappushpop(heap, (score, doc))

        # step 3: get the topK docs from the heap
        heap = heapq.nhighest(topK, heap)

        # step 4: return the topK docs
        return heap

    """ Get the query vector based on the postings_lists

    Args:
        terms: a dict contains all the terms and their number of occurrences
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        query_vector: the query vector based on VSM
    """
    def _get_query_vector(self, terms, postings_lists):
        pass

    """ Get the document vector based on the postings_lists

    Args:
        doc: the doc id where the document vector needs to be calculate
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        doc_vector: the document vector based on VSM
    """
    def _get_doc_vector(self, doc, postings_lists):
        pass

    """ Get the universe of the docs that appear in one of the lists.

    Args:
        terms: a dict contains all the terms and their number of occurrences
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
        terms: a dict contains all the terms and their number of occurrences
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        intersection: the intersection of postings lists
    """
    def _get_intersection(self, terms, postings_lists):
        # optimize the order of the merge
        costs = []
        for term in terms:
            postings = postings_lists[term][0]
            costs.append((term, postings.shape[0]))

        costs.sort(key = lambda key: key[1])

        # perform pairwise merge
        result = postings_lists[costs[0][0]][0][:,0]
        for i in range(1, len(costs)):
            term = costs[i][0]
            postings = postings_lists[term][0]

            p1 = p2 = 0
            len1, len2 = len(result), len(postings)
            temp = array.array('i')

            while p1 < len1 and p2 < len2:
                doc1 = result[p1]
                doc2 = postings[p2][0]

                if doc1 == doc2:
                    temp.append(doc1)
                    p1, p2 = p1+1, p2+1
                elif doc1 < doc2:
                    p1 += 1
                else:
                    p2 += 1

            result = temp

        # return the intersection
        result = list(result)

        return result

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
        terms: a dict contains all the terms and their number of occurrences
        tokens: a list contains all the tokens appeared in the query string
    """
    def _tokenize(self, query):
        # tokenize the query string
        tokens = [word for sent in nltk.sent_tokenize(query)
                       for word in nltk.word_tokenize(sent)]

        # stem the tokens
        tokens = [self.stemmer.stem(token).lower() for token in tokens]

        # get the dict of tokens
        terms = defaultdict(lambda: 0)
        for token in tokens:
            terms[token] += 1

        return terms, tokens

if __name__ == '__main__':
    # Create a Searcher
    searcher = Searcher('dictionary.txt', 'postings.txt', phrasal = False, pivoted = False)

    terms = ['into', 'queri', 'can', 'term', 'searcher', 'token', 'string', 'and']
    postings_lists = {
        'into'    : (np.array([[0, 1], [1, 5], [3, 6], [5, 1]]), ),
        'queri'   : (np.array([[0, 5]]), ),
        'can'     : (np.array([[0, 1], [7, 10], [9, 3]]), ),
        'term'    : (np.array([[0, 1], [2, 5], [4, 6], [6, 10]]), ),
        'searcher': (np.array([[0, 1], [8, 3]]), ),
        'token'   : (np.array([[0, 1], [1, 7], [4, 6], [7, 3]]), ),
        'string'  : (np.array([[0, 1], [2, 5], [5, 6], [8, 1]]), ),
        'and'     : (np.array([[0, 1], [6, 6], [9, 9]]), )
    }

    test = '_get_intersection'
    # Test tokenizing the query string
    if test == '_tokenize':
        terms, tokens = searcher._tokenize('Searcher can tokenize query strings into terms and tokens')
        print(terms)
        print(tokens)
    elif test == '_get_universe':
        universe = searcher._get_universe(terms, postings_lists)
        print(universe)
    elif test == '_get_intersection':
        intersection = searcher._get_intersection(terms, postings_lists)
        print(intersection)
