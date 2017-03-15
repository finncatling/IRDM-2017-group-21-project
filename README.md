# irdm
Information retrieval and data mining assignment

### files/directories

This is how the pre-processing.py and analysis.py files refer to the directories. 

![Alt text](./misc/directories.png?raw=true "Optional Title")

- irdm is the cloned repository.
- the data files/directory are local.
- funcs.py contains abstracted functions used in pre-processing.py.
- funcs.py requires your nltk version to be less than 3.2.2
- pre-processing.py take ~11m to run on my 2.2 GHz Intel Core i7. It outputs a csv file (457Mb) and a .pickle file (382Mb). 
- analysis.py loads the pickle file, ready to train your models.

