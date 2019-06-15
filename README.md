## Personalized Medicine: Redefining Cancer Treatment

------
### Usage
1. Clone the repo

```
git clone git@github.com:dujm/Health_PrecisionMedicine.git

# Remove my git directory
cd Health_PrecisionMedicine
rm -r .git/

# Create a src/ dir to save models
mkdir src

# Create a data/ to download and save datasets
mkdir data
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
    
    # Download data
    kaggle competitions download msk-redefining-cancer-treatment
    ```

------
###  Files

     ├── LICENSE
     ├── README.md
     ├── requirements.txt   
     ├── data (Not uploaded to GitHub)
     │   ├── features
     │   ├── history
     │   ├── processed
     │   └── raw
     │       ├── test_text
     │       ├── test_variants
     │       ├── training_text
     │       ├──training_variants
     │       └── not_used
     ├── notebooks
     │   ├── 01EDA.ipynb
     │   ├── 02Test_Sample_Data.ipynb
     │   ├── 03BoW_Full_Data.ipynb
     │   ├── 04Word2Vec_LSTM_Full_Data.ipynb
     │   ├── 05Doc2Vec_Keras_Full_Data.ipynb  
     │   └── utils_functions.py
     ├── reports
     │   ├── figures
     │   └── materials
     ├── src (Not uploaded to GitHub)
     │   ├── doc2vec
     └── └── word2vec




------
### References
 * [Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment)
 * [A simple anaysis of the dataset using nltk and Word2Vec](https://www.kaggle.com/umutto/preliminary-data-analysis-using-word2vec/data)
 * [Basic NLP](https://www.kaggle.com/reiinakano/basic-nlp-bag-of-words-tf-idf-word2vec-lstm)
 * Paweł Jankiewicz
