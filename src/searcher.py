#!/usr/bin/python3

import sys
import math
import numpy as np

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
        pass


    """ Search and return docIds according to the boolean expression.

    Args:
        query: the query string

    Returns:
        result: the list of 10 most relevant docIds in response to the query
    """
    def search(self, query):
        # step 1: tokenize the query to get all the terms

        # step 2: get the postings lists of the terms

        # text query:
        # Step 3: get total relevant docIds

        # phrasal query:
        # step 3-1: get all the docs that contains all the terms in the query
        # step 3-2: judging every doc whether is contains the phrase

        # step 4: pass the docs to the rank function get the result

        # step 5: return the result
        pass

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
        pass

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
        terms: a list contains all the terms appeared in the boolean expression
    """
    def _tokenize(self, query):
        pass
