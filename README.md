## Personalized Medicine: Redefining Cancer Treatment

------
### Usage
1. Clone the repo

```
git clone git@github.com:dujm/Health_PrecisionMedicine.git

# Remove my git directory
cd Health_PrecisionMedicine
rm -r .git/
```

2. Install packages

```
pip install -r requirements.txt
```

3. Download the Kaggle dataset
 * [Download from Kaggle Website](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data)

 * [Or install Kaggle API](https://dujm.github.io/datasciences/kaggle) and run:

    ```
    cd data
    kaggle competitions download msk-redefining-cancer-treatment
    ```

------
###  Files

     ├── LICENSE
     ├── README.md
     ├── data
     │   ├── external
     │   ├── interim
     │   ├── processed
     │   │   ├── class_corpus.txt
     │   │   ├── test_variants_text.csv
     │   │   └── train_variants_text.csv
     │   └── raw
     │       ├── test_text
     │       ├── test_variants
     │       ├── training_text
     │       ├──training_variants
     │       └── not_used
     ├── models
     ├── notebooks
     │   ├── 01EDA.ipynb
     │   ├── 02Feature_and_Model_Sample_Data.ipynb
     │   ├── 03Feature_Model_Full_Data.ipynb
     │   └── utils_functions.py
     ├── references
     ├── reports
     │   ├── figures
     │   └── materials
     ├── requirements.txt
     └──src


------
### References
 * [Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment)
 * [A simple anaysis of the dataset using nltk and Word2Vec](https://www.kaggle.com/umutto/preliminary-data-analysis-using-word2vec/data)
 * [Basic NLP](https://www.kaggle.com/reiinakano/basic-nlp-bag-of-words-tf-idf-word2vec-lstm)
 * Paweł Jankiewicz
