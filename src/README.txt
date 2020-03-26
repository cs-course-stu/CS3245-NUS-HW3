This is the README file for A0214251W-A0000000X's submission

== Python Version ==

We're using Python Version 3.7.4 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps
in your program, and discuss your experiments in general.  A few paragraphs
are usually sufficient.

We divide the whole program into two parts, the construction part and the search
part. The former is to build the inverted index, record the tfs corresponding
to the postings and save the index into file. If the phrase query is enabled, the indexer
will also save the term postions in the doc. The searcher is to load the inverted index,
parse the query, search the K most relevant documents in repsonse to the query.
We adopt the object-oriented programming approach to build the whole project, which greatly
reduces the coupling between two parts.

[Arguments]
The program provides additional command line arguments to support more features.
  Details as follows:
    -d  dictionary file path
    -p  postings file path
    -q  queries file path
    -o  search results file path
    -t  number of the results returned by each query
    -r  penalty rate of the pivoted normalized document length
    -x  enable phrasal query
    -n  enable using pivoted normalized document length
    -s  enable printing score

  Examples:
    python index.py -i Path-to-Reuters -d dictionary.txt -p postings.txt -n -x
    python search.py -d dictionary.txt -p postings.txt -q queries1.txt -o results1.txt -t 10 -r 0.01 -x -n -s


[Indexer]

[Searcher] A0214251W
As for the search part, we use the following steps to process a request.
    1. Tokenize the query into tokens, the program will also count the number of each term
       in the query.
    2. Load postings lists for the terms in the expression from the postings.txt file.
       For terms that appear multiple times in the query, we just load once to reduce memory cost.
       The postings lists include the docids, tfs and positions if the phrase query is turned on.
    3. Get the docs that need to rank.
       This step is mainly used to filter documents when phrase query is turned on.

       [phrasal query filtering]
       3.1 Get all the docs that contains all the terms inthe query.
           Calculate the union of all postings lists, the searcher will merge them
           according to the size of the list.
       3.2 Judging every doc whether it contains the phrase.
           After step 3.1, the candidate docs contain all terms but they may not adjacent.
           So we need to ensure the candidate docs contain the phrase.
    4. Rank the candidate docs and get the result.

       4.1 Construct the query vector
           Based on the postings lists of terms, we can get the query vector.
       4.2 Processing every document and every term.
           To get the cosine value of document vector and the query vector as doc's score.
       4.3 Divide the score with the document length
           If the pivoted normalized document length is turned on, the score will be devided by
           the pivoted normalized document length.
       4.4 Use max-head heap to get the K(can be set by user) most relevant docIds.
           If printing score is enabled, the scores will be returned too.
    5. Return the results

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

* indexer.py: The file contains the Indexer class which helps to build, load and dump the inverted index.
* searcher.py: The file contains the Searcher class which helps to search the K most relevant documents of the query.
* index.py: The file in this assignment using Indexer to build the index and dump to file.
* search.py: The file in this assignment using Seacher to get the top K documents.
* README.txt: This file contains this sentence gives an overview of this assignment.
* ESSAY.txt: This file contains essay questions mentioned on HW#3

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] I/We, A0214251W-A0000000X, certify that I/we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I/we
expressly vow that I/we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

* Complete all the requirements.
* Good programming habits, clear comments.
* Support additional command line arguments which is convenient to start various tests.
* Support phrasal query and pivoted normalized document length.
* Good performance:
  * Avoid loading same term multiple times to save time and space.
  * Use heap to get the top K documents.
  * Speed up merge when phrase is turned on.
  * When phrase query is turned on, the intersection of all lists is performed first to narrow down the scope


<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>

* [To search some Python functions](https://devdocs.io/python/)
* [Numpy Reference](https://docs.scipy.org/doc/numpy/reference/)
* [NLTK API](https://www.nltk.org/api/nltk.html)
