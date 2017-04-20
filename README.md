# IRDM-2017-group-21-project
COMPGI15: Information Retrieval and Data Mining - Home Depot Kaggle Challenge

### Running the code
This is how the pre-processing.py and analysis.py files refer to the directories. 

![Alt text](./misc/directories.png?raw=true "Optional Title")

- irdm is the cloned repository.
- the data files/directory are local.
- spelling.py contain the spelling error dictionary
- funcs.py contains abstracted functions used in pre-processing.py.
- funcs.py requires your nltk version to be less than 3.2.2
- pre-processing.py take ~11m to run on my 2.2 GHz Intel Core i7. It outputs a csv file (457Mb) and a .pickle file (382Mb). 
- feature_anova_f.py generates the ANOVA F scores for the features.
- analysis.py loads the pickle file, ready to train your models.
- analysis_xgb.py loads the pickle file, finds optimal parameters for the boosted trees model, and calculates public and private test scores for the optimal model
- analysis_sv.py loads the pickle file, finds optimal parameters for the SVR models, and calculates public and private test scores for the optimal model. Please edit for SVC model.
- create_vocab.py creates a vocabulary for running neural nets out of search_term, product_title, product_description and brand name (extracted feature) features. It loads pickle file which is generated from pre_processing.py file with switch stem = false. Location of the file to load is https://drive.google.com/open?id=0Bylsb5Tv26G-SzE4S0xzMHEtZmM
- Vocab_ib_GloVe.py converts vocabulary word values to numeric vectors in GloVe.
- nnets_model.py uses vocabulary from Vocab_ib_GloVe.py to feed neural nets and then runs simple feed forward network with one hidden layer.
