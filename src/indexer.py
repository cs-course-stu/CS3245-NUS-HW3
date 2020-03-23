import numpy as np
import os
import sys
import nltk
import pickle
import math
import time
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import OrderedDict


class Indexer:
    dictionary = {}
    """ class InvertedIndex is a class dealing with building index, saving it to file and loading it

    Args:
        dictionary_file: the name of the dictionary.
        postings_file: the name of the postings
        phrasal_query: Boolean varible of phrase query searching
        normalize: Boolean varible of advanced normaliza according to the ESSAY #2
    """

    def __init__(self, dictionary_file, postings_file, phrasal_query, normalize):
        self.dictionary_file = dictionary_file
        self.postings_file = postings_file
        self.total_doc = {}
        self.dictionary = {}
        self.skip_pointer_list = []
        self.postings = {}
        """
        non-position (non -x):
        test1 = {"term": [[1,2],[5,6]]}
        position (-x):
        test1 = {"term": [[1,2,[3,4]],[5,6,[7,8]]]}
        """
        self.file_handle = None
        self.average = 0
        self.phrasal_query = phrasal_query
        self.normalize = normalize

    """ build index from documents stored in the input directory

    Args: 
        in_dir: working path
    """

    def build_index(self, in_dir):

        print('indexing...')
        if (not os.path.exists(in_dir)):
            print("wrong file path!")
            sys.exit(2)
        files = os.listdir(in_dir)
        porter_stemmer = PorterStemmer()

        for i, file in enumerate(files):
            # if i > 100:
            #     break
            if not os.path.isdir(file):
                doc_id = int(file)
                doc_set = set()
                term_pos = 0
                self.total_doc[doc_id] = 0
                f = open(in_dir+"/"+file)
                for line in iter(f):
                    # tokenize
                    tokens = [word for sent in nltk.sent_tokenize(
                        line) for word in nltk.word_tokenize(sent)]

                    for token in tokens:
                        # stemmer.lower
                        clean_token = porter_stemmer.stem(token).lower()

                        if clean_token in self.dictionary:  # term exists
                            if clean_token in doc_set:
                                self.postings[clean_token][-1][1] += 1

                                # insert position
                                if(self.phrasal_query):
                                    self.postings[clean_token][-1][2].append(
                                        term_pos)

                            else:
                                doc_set.add(clean_token)

                                # insert position
                                if(self.phrasal_query):
                                    self.postings[clean_token].append(
                                        [doc_id, 1, [term_pos]])
                                else:
                                    self.postings[clean_token].append(
                                        [doc_id, 1])
                        else:
                            doc_set.add(clean_token)
                            self.dictionary[clean_token] = 0

                            # insert position
                            if(self.phrasal_query):
                                self.postings[clean_token] = [
                                    [doc_id, 1, [term_pos]]]  # {"term": [[1,2],[5,6]]}
                            else:
                                self.postings[clean_token] = [
                                    [doc_id, 1]]  # {"term": [[1,2],[5,6]]}
                        term_pos += 1
                # accumulate the length of doc
                if(self.normalize):
                    self.average += term_pos

                # calculate weight of each term
                length = 0
                for token in doc_set:
                    # tf of each term
                    length += np.square(1 +
                                        math.log(self.postings[token][-1][1], 10))

                # sqart the length and assign it to doc
                self.total_doc[doc_id] = np.sqrt(length)

        # calculate the average length of doc
        if(self.normalize):
            self.average /= (i+1)

        print('build index successfully!')

    """ save dictionary, postings and skip pointers given fom build_index() to file

    """

    def SavetoFile(self):
        print('saving to file...')

        dict_file = open(self.dictionary_file, 'wb+')
        post_file = open(self.postings_file, 'wb+')
        pos = 0

        # save postings to the file
        for key, value in self.postings.items():
            # save the offset of dictionary
            pos = post_file.tell()
            self.dictionary[key] = pos

            # print(self.postings[key])
            tmp = np.array(self.postings[key], dtype=object)

            # operate each postings
            for i in range(len(tmp)):
                # convert tf to 1 + log() format
                # tmp[i][1] = 1 + \
                #     math.log(tmp[i][1], 10)

                # convert position list to the np.array
                if(self.phrasal_query):
                    tmp[i][2] = np.array(tmp[i][2])

            # sort the posting list according to he doc_id
            tmp = tmp[tmp[:, 0].argsort()]
            # print(tmp)

            # split the total postings into doc_plus_tf and position list
            doc_plus_tf = np.array(tmp[:, 0:2], dtype=np.int32)
            if(self.phrasal_query):
                position = np.array(tmp[:, 2])
            # print(doc_plus_tf)
            # print(position)

            # save all content to the file
            np.save(post_file, doc_plus_tf, allow_pickle=True)
            if(self.phrasal_query):
                np.save(post_file, position, allow_pickle=True)

        # save average length of doc to the file 0 means null
        pickle.dump(self.average, dict_file)

        # save total_doc and dictionary
        pickle.dump(self.total_doc, dict_file)
        pickle.dump(self.dictionary, dict_file)

        print('save to file successfully!')
        return

    """ load dictionary from file

    Returns:
        total_doc: total doc_id
        dictionary: all word list
    """

    def LoadDict(self):
        print('loading dictionary...')
        with open(self.dictionary_file, 'rb') as f:
            self.average = pickle.load(f)
            self.total_doc = pickle.load(f)
            self.dictionary = pickle.load(f)

        self.file_hande = open(self.postings_file, 'rb')

        print('load dictionary successfully!')
        return self.average, self.total_doc, self.dictionary

    """ load multiple postings lists from file

    Args:
        terms: the list of terms need to be loaded

    Returns:
        postings_lists: the postings lists correspond to the terms
    """

    def LoadTerms(self, terms):
        if not self.file_handle:
            self.file_handle = open(self.postings_file, 'rb')

        ret = {}
        for term in terms:
            if term in self.dictionary:
                self.file_handle.seek(self.dictionary[term])
                # load postings and position
                doc_plus_tf = np.load(self.file_handle, allow_pickle=True)
                if(self.phrasal_query):
                    position = np.load(
                        self.file_handle, allow_pickle=True).tolist()
                    ret[term] = (doc_plus_tf, position)
                else:
                    ret[term] = (doc_plus_tf, )
                # for i in range(len(doc_plus_tf)):
                #     tmp = position[i]
                #     print(tmp)
                    # print(np.array([doc_plus_tf[i], position[i]]).tolist())
                # print(len(doc_plus_tf))
                # print(position[0])
                # print(np.array([doc_plus_tf[0], position[0]]))

                # data = np.concatenate([doc_plus_tf, position], axis=1)
                # postings = np.load(self.file_handle, allow_pickle=True)
                # pointers = self.skip_pointer_list[len(postings)]
            else:
                doc_plus_tf = np.empty(shape=(0, 2), dtype=np.int32)
                if self.phrasal_query:
                    position = np.empty(shape=(0, ), dtype=object)
                    ret[term] = (doc_plus_tf, position)
                else:
                    ret[term] = (doc_plus_tf, )

        return ret

if __name__ == '__main__':
    indexer = Indexer(
        'dictionary.txt', 'postings.txt', phrasal_query=True, normalize=True)
    #indexer.build_index('../../reuters/training')
    # start = time.time()
    indexer.build_index(
        '/Users/wangyifan/Google Drive/reuters/training')
    indexer.SavetoFile()
    # end = time.time()
    # print('execution time: ' + str(end-start) + 's')
    average, total_doc, dictionary = indexer.LoadDict()
    terms = ['of']
    print(indexer.LoadTerms(terms))
