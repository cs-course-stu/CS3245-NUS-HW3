#!/usr/bin/python3

import sys
import math
import nltk
import array
import heapq
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

    """ Search and return docIds according to the boolean expression.

    Args:
        query: the query string

    Returns:
        result: the list of 10 most relevant docIds in response to the query
    """
    def search(self, query):
        # step 1: tokenize the query to get all the terms
        terms, counts, tokens = self._tokenize(query)

        # step 2: get the postings lists of the terms
        postings_lists = self.indexer.LoadTerms(terms)

        # step 3: get the docs that need to rank
        if self.phrasal:
            # phrasal query
            # step 3-1: get all the docs that contains all the terms in the query
            candidate = self._get_intersection(terms, postings_lists)

            # step 3-2: judging every doc whether it contains the phrase
            candidate = self._judge(candidate, tokens, postings_lists)
            pass
        else:
            # text query
            candidate = []

        # step 4: pass the condidate docs to the rank function get the result
        result = self.rank(terms, counts, candidate, postings_lists)

        # step 5: return the result
        return result

    """ Rank the documents and return the 10 most relevant docIds.
        The result should be in the order of relevant.

    Args:
        terms: all terms in the query string
        counts: number of occurrences of all terms
        doc_list: the list of candidate docs
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
       result: the list of 10 most relevant docIds in response to the query
    """
    def rank(self, terms, counts, doc_list, postings_lists):
        # step 1: initialze the scores
        scores = defaultdict(lambda: 0)

        # step 2: construct the query vector
        query_vector = self._get_query_vector(terms, counts, postings_lists)

        # step 3: processing every document and every term
        for i, term in enumerate(terms):
            postings = postings_lists[term][0]
            for j in range(0, len(postings)):
                doc = postings[j][0]

                if self.phrasal and (doc not in doc_list):
                    continue

                weight = postings[j][1]
                scores[doc] += weight * query_vector[i]

        # step 4: get the topK docs from the heap
        heap = [(scores[doc], doc) for doc in scores]
        heap = heapq.nlargest(self.topK, heap)

        # step 5: return the topK docs
        return heap

    """ Get the query vector based on the postings_lists

    Args:
        terms: all terms in the query string
        counts: number of occurrences of all terms
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        query_vector: the query vector based on VSM
    """
    def _get_query_vector(self, terms, counts, postings_lists):
        total_num = len(self.total_doc)
        query_vector = np.zeros(len(terms))

        length = 0
        for i, term in enumerate(terms):
            tf = 1 + math.log(counts[i])
            df = len(postings_lists[term][0])
            idf = math.log(total_num / df) if df else 0
            weight = tf * idf

            query_vector[i] = weight
            length += weight * weight

        length = math.sqrt(length)

        for i in range(0, len(terms)):
            query_vector[i] /= length

        return query_vector

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
                    p1, p2 = p1 + 1, p2 + 1
                elif doc1 < doc2:
                    p1 += 1
                else:
                    p2 += 1

            result = temp

        # return the intersection
        result = list(result)

        return result

    """ Judging whether candidate documents contain the phrase

    Args:
        candidate: candidate documents to be ranked
        tokens: a list contains all the tokens appeared in the query string
        postings_lists: the dictionary with terms to posting lists mapping

    Returns:
        ans: the final candidate documents
    """
    def _judge(self, candidate, tokens, postings_lists):
        if len(tokens) <= 1:
            return candidate

        positions = defaultdict(lambda: [])
        candidate = set(candidate)

        # get postions for docs
        for i, token in enumerate(tokens):
            postings_list = postings_lists[token]
            postings = postings_list[0]
            length = postings.shape[0]
            for j in range(0, length):
                docId = postings[j][0]
                if docId in candidate:
                    positions[docId].append(postings_list[1][j])

        # judging every doc
        ans = []
        for doc in positions:
            position = positions[doc]
            pointers = [0] * len(position)

            index = 1
            flag = False
            prev_pos = position[0][0]
            while True:
                pointer = pointers[index]
                length = len(position[index])

                while pointer + 1 < length:
                    tmp = position[index][pointer + 1]
                    if tmp <= prev_pos + 1:
                        pointer += 1
                    else:
                        break

                pointers[index] = pointer
                cur_pos = position[index][pointer]

                if cur_pos != prev_pos + 1:
                    index -= 1
                    pointers[index] += 1
                    if pointers[index] >= len(position[index]):
                        break
                    if index == 0:
                        index += 1

                    pointer = pointers[index - 1]
                    prev_pos = position[index - 1][pointer]
                    continue
                else:
                    prev_pos = cur_pos
                    index += 1
                    if index >= len(position):
                        flag = True
                        break

            if flag:
                ans.append(doc)

        return ans

    """ Tokenize the query into single term.

    Args:
        query: the query string

    Returns:
        terms: all terms in the query string
        counts: number of occurrences of all terms
        tokens: a list contains all the tokens appeared in the query string
    """
    def _tokenize(self, query):
        # tokenize the query string
        tokens = [word for sent in nltk.sent_tokenize(query)
                       for word in nltk.word_tokenize(sent)]

        # stem the tokens
        tokens = [self.stemmer.stem(token).lower() for token in tokens]

        # get the term count
        term_count = defaultdict(lambda: 0)
        for token in tokens:
            term_count[token] += 1

        # get terms and counts
        terms = []
        counts = []
        for term in term_count:
            terms.append(term)
            counts.append(term_count[term])

        return terms, counts, tokens

if __name__ == '__main__':
    # Create a Searcher
    searcher = Searcher('dictionary.txt', 'postings.txt', phrasal = False, pivoted = False)

    query = 'Searcher can tokenize query strings into terms and tokens'
    terms = ['searcher', 'can', 'token', 'queri', 'string', 'into', 'term', 'and']
    counts = [1, 1, 2, 1, 1, 1, 1, 1]
    postings_lists = {
        'into'    : (np.array([[0, 1], [1, 5], [3, 6], [5, 1]]), [np.array([5, ])]),
        'queri'   : (np.array([[0, 5]])                        , [np.array([3, ])]),
        'can'     : (np.array([[0, 1], [7, 10], [9, 3]])       , [np.array([1, ])]),
        'term'    : (np.array([[0, 1], [2, 5], [4, 6], [6, 10]]),[np.array([6, ])]),
        'searcher': (np.array([[0, 1], [8, 3]])                , [np.array([0, ])]),
        'token'   : (np.array([[0, 1], [1, 7], [4, 6], [7, 3]]), [np.array([2, 8])]),
        'string'  : (np.array([[0, 1], [2, 5], [5, 6], [8, 1]]), [np.array([4, ])]),
        'and'     : (np.array([[0, 1], [6, 6], [9, 9]]),         [np.array([7, ])])
    }

    test = 'search'

    # Tests
    if test == '_tokenize':
        terms, counts, tokens = searcher._tokenize(query)
        print(terms)
        print(counts)
        print(tokens)
    elif test == '_get_universe':
        universe = searcher._get_universe(terms, postings_lists)
        print(universe)
    elif test == '_get_intersection':
        intersection = searcher._get_intersection(terms, postings_lists)
        print(intersection)
    elif test == '_get_query_vector':
        query_vector = searcher._get_query_vector(terms, counts, postings_lists)
        print(query_vector)
    elif test == 'search':
        result = searcher.search('share in quarter and')
        print(result)
    elif test == '_judge':
        terms.append('token')
        result = searcher._judge([0], terms, postings_lists)
        print(result)
