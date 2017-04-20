# irdm
Information retrieval and data mining assignment

### pre-processed data

- 15/4/17, including punkt tokenisation, stopword and punctuation removal, stemming, cosine similarity, bm25, shuffled training data: https://we.tl/E7vIV6y8kD
- 15/4/17, including punkt tokenisation, punctuation removal **but stopwords left in**, stemming, cosine similarity, bm25, shuffled training data: https://we.tl/JMcSxVLVAg
- 18/4/17, including all the features as above files have, except for bm25, stopword and punctuation removal; **but no stemming** https://drive.google.com/open?id=0Bylsb5Tv26G-SzE4S0xzMHEtZmM - this is mainly for GloVe
- 18/4/17, including vocabulary words from search term, product description, product title and brand name **no stemming** https://drive.google.com/open?id=0Bylsb5Tv26G-Mk5hWjFjSldIaXM
- 20/4/17, "dl_full_size.pickle" - file that contains only 4 features needed to run nnets_model.py https://drive.google.com/open?id=0Bylsb5Tv26G-Q01qc1REelhsYlU 



### files/directories

This is how the pre-processing.py and analysis.py files refer to the directories. 

![Alt text](./misc/directories.png?raw=true "Optional Title")

- irdm is the cloned repository.
- the data files/directory are local.
- funcs.py contains abstracted functions used in pre-processing.py.
- funcs.py requires your nltk version to be less than 3.2.2
- pre-processing.py take ~11m to run on my 2.2 GHz Intel Core i7. It outputs a csv file (457Mb) and a .pickle file (382Mb). 
- analysis.py loads the pickle file, ready to train your models.

