# -*- coding: utf-8 -*-
# Load packages
# System packages
import os
import re
import datetime
import warnings
warnings.simplefilter("ignore", UserWarning)

# Data related
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns, matplotlib.pyplot as plt

# Text analysis helper libraries
from gensim.summarization import summarize, keywords
from gensim.models import KeyedVectors

# +
# Text analysis helper libraries for word frequency
import nltk
# Download stop words
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import snowballstemmer



# -


# Word cloud visualization libraries
from scipy.misc import imresize
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter


# +
# Dimensionaly reduction libraries
from sklearn.decomposition import PCA

# Clustering library
from sklearn.cluster import KMeans
# -

# sklearn 
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, accuracy_score
import scikitplot.plotters as skplt


# Create a new folder function
def createFolder(directory):
    '''
    Create a new folder
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# Data munging function
def dm(data):
    '''
    Summarize data column features in a new data frame
    '''
    unique_data = pd.DataFrame(columns =('colname','dtype','Null_sum','unique_number','unique_values'))
    for col in data:
        if data[col].nunique() <25:
            unique_data = unique_data.append({'colname': col, \
                                              'dtype': data[col].dtype,\
                                              'Null_sum':data[col].isnull().sum(),\
                                              'unique_number': data[col].nunique(),\
                                              'unique_values':data[col].unique()}, \
                                     ignore_index=True)
        else:
            unique_data = unique_data.append({'colname': col, \
                                              'dtype': data[col].dtype,\
                                              'Null_sum':data[col].isnull().sum(),\
                                              'unique_number': data[col].nunique(),\
                                              'unique_values':'>25'}, \
                                     ignore_index=True)
    return unique_data.sort_values(by=['unique_number','dtype'])


# Group by a column and count the size 
def groupby_col_count(df, colname,save_csv_dir=None):
    df1 = df.groupby([df[colname]]) \
        .size() \
        .sort_values(ascending=False)
    name= 'groupby_' +str(colname)
    csvname = '{}.csv'.format(os.path.join(save_csv_dir,name))
    df1.to_csv(csvname,index=False)
    return df1.head(5)


# Bar Plot: Count by colname
def col_count_plot(df,colname,save_plot_dir=None):
    '''
    df: dataframe
    colname: column name
    save_plot_dir: saving plot directory
    '''
    df=df.drop_duplicates()
    number = df[colname].value_counts().values
    number = [str(x) for x in number.tolist()]
    number = ['n: ' + i for i in number]
    ax = sns.countplot(x=colname,  data=df)
    pos = range(len(number))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick], + 0.1, number[tick], horizontalalignment='center', size='small', color='w', weight='semibold')
    fig = ax.get_figure()
    #save plot
    name= str(colname)+'count'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    fig.savefig(figname, figdpi = 300)


# Frequency plot of a col
def frequency_plot(df,colname,save_plot_dir=None):
    '''
    df: dataframe
    colname: column name
    save_plot_dir: saving plot directory
    '''
    plt.figure()
    ax = df[colname].value_counts().plot(kind='area')
    ax.get_xaxis().set_ticks([])
    ax.set_title('Train Data: ' + str(colname) +' Frequency Plot')
    ax.set_xlabel(colname)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    #save plot
    name= str(colname)+'_frequency'
    plotname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    plt.savefig(plotname, figdpi = 300)
# Resize an image
def resize_image(np_img, new_size):
    old_size = np_img.shape
    ratio = min(new_size[0]/old_size[0], new_size[1]/old_size[1])

    return imresize(np_img, (round(old_size[0]*ratio), round(old_size[1]*ratio)))

# Get average vector from text
def get_average_vector(text,stop_words):
    tokens = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_words]
    return np.mean(np.array([model.wv[w] for w in tokens if w in model]), axis=0)

custom_words = ["fig", "figure", "et", "al", "al.", "also",
                "data", "analyze", "study", "table", "using",
                "method", "result", "conclusion", "author", 
                "find", "found", "show", '"', "’", "“", "”"]
stop_words = set(stopwords.words('english') + list(punctuation) + custom_words)


# Build a corpus for a Text column grouped by Target columns
def build_corpus(df,target,text,stop_words,wordnet_lemmatizer):
    '''
    df: dataframe
    target: prediction target column
    text: text column
    '''
    class_corpus = df.groupby(target).apply(lambda x: x[text].str.cat())
    class_corpus = class_corpus.apply(lambda x: Counter([wordnet_lemmatizer.lemmatize(w)
    for w in word_tokenize(x) if w.lower() not in stop_words and not w.isdigit()]
    # Save the corpus
    #class_corpus.to_csv('../data/processed/class_corpus.txt',sep='\t',index=False)
    ))
    return class_corpus


# World frequency plot
def word_cloud_plot_no_mask(corpus,save_plot_dir=None):
    whole_text_freq = corpus.sum()
    wc = WordCloud(max_font_size=300,min_font_size=30,
               max_words=1000,
               width=4000,
               height=2000,
               prefer_horizontal=.9,
               relative_scaling=.52,
               background_color='black',
               mask=None,
               mode="RGBA").generate_from_frequencies(whole_text_freq)
    plt.figure()
    plt.axis("off")
    plt.tight_layout()
    #plt.savefig(figname, figdpi = 300)
    plt.imshow(wc, interpolation="bilinear")
    #save plot
    #plt.figure(figsize=(10,5))
    name= 'word_cloud_plot'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    plt.savefig(figname,figdpi = 600)
    plt.show()
    plt.close()

# Build a word cloud without mask image
def word_cloud_plot_no_mask(corpus,save_plot_dir=None):
    whole_text_freq = corpus.sum()
    wc = WordCloud(max_font_size=300,min_font_size=30,
               max_words=1000,
               width=4000,
               height=2000,
               prefer_horizontal=.9,
               relative_scaling=.52,
               background_color='black',
               mask=None,
               mode="RGBA").generate_from_frequencies(whole_text_freq)
    plt.figure()
    plt.axis("off")
    plt.tight_layout()
    #plt.savefig(figname, figdpi = 300)
    plt.imshow(wc, interpolation="bilinear")
    #save plot
    #plt.figure(figsize=(10,5))
    name= 'word_cloud_plot'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    plt.savefig(figname,figdpi = 300)
    plt.close()

# Build a word cloud with mask image
def word_cloud_plot(mask_image_path, corpus,save_plot_dir):
    mask_image = np.array(Image.open(mask_image_path).convert('L'))
    mask_image = resize_image(mask_image, (8000, 4000))
    whole_text_freq = corpus.sum()
    wc = WordCloud(max_font_size=300,min_font_size=30,
               max_words=1000,
               width=mask_image.shape[1],
               height=mask_image.shape[0],
               prefer_horizontal=.9,
               relative_scaling=.52,
               background_color=None,
               mask=mask_image,
               mode="RGBA").generate_from_frequencies(whole_text_freq)
    plt.figure()
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(wc, interpolation="bilinear")
    #save plot
    name= 'word_cloud_mask_plot'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    plt.savefig(figname, figdpi = 600)


# PCA plot
def pca_plot(classes, vecs,save_plot_dir=None):
    pca = PCA(n_components=2)
    reduced_vecs = pca.fit_transform(vecs)
    fig, ax = plt.subplots()
    cm = plt.get_cmap('jet', 9)
    colors = [cm(i/9) for i in range(9)]
    ax.scatter(reduced_vecs[:,0], reduced_vecs[:,1], c=[colors[c-1] for c in classes], cmap='jet', s=8)
    # adjust x and y limit
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    plt.legend(handles=[Patch(color=colors[i], label='Class {}'.format(i+1)) for i in range(9)])
    plt.show()
    #save plot
    name= 'pca_plot'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    #image
    fig.savefig(figname, figdpi = 300)
    plt.close()

# kmeans plot
def kmeans_plot(classes, vecs,save_plot_dir=None):
    kmeans = KMeans(n_clusters=9).fit(vecs)
    c_labels = kmeans.labels_
    fig, ax = plt.subplots()
    cm = plt.get_cmap('jet', 9)
    colors = [cm(i/9) for i in range(9)]
    ax.scatter(reduced_vecs[:,0], reduced_vecs[:,1], c=[colors[c-1] for c in c_labels], cmap='jet', s=8)
    plt.legend(handles=[Patch(color=colors[i], label='Class {}'.format(i+1)) for i in range(9)])
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    plt.show()
    #save plot
    name= 'kmeans_plot'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    #image
    fig.savefig(figname, figdpi = 300)
    plt.close()


def evaluate_features(X, y, clf):
    """General helper function for evaluating effectiveness of passed features in ML model
    
    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation
    
    Args:
        X (array-like): Features array. Shape (n_samples, n_features)
        
        y (array-like): Labels array. Shape (n_samples,)
        
        clf: Classifier to use. If None, default Log reg is use.
        e.g.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import log_loss, accuracy_score
        from sklearn.datasets import load_iris
        clf = LogisticRegression()
        evaluate_features(*load_iris(True),clf)
    """
    if clf is None:
        pass
    else:
        probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8), \
                                   n_jobs=-1, method='predict_proba', verbose=2)
        pred_indices = np.argmax(probas, axis=1)
        classes = np.unique(y)
        preds = classes[pred_indices]
        print('Log loss: {}'.format(log_loss(y, probas)))
        print('Accuracy: {}'.format(accuracy_score(y, preds)))
        skplt.plot_confusion_matrix(y, preds)


# Split a pandas dataframe into train and validation dataset 
def split_data(df,text,target,test_size,random_state,stratify=None):
    '''
    df[text]: text data for training
    df[target]: label of text data
    
    '''
    df[text] = df[text].astype(str)
    df[target] = df[target].astype(str)
    X= df[text].values
    y =df[target].values
    X_tr, X_val, y_tr, y_val = train_test_split(X,
                                            y,
                                            test_size=test_size,
                                            stratify=df[target],
                                            random_state=random_state)
    return X_tr, X_val, y_tr, y_val

# +


def clean_text(t):
    """Accepts a Document 
    """
    t = t.lower()
    # Remove single characters
    t = re.sub("[^A-Za-z0-9]"," ",t)
    # Replace all numbers by a single char
    t = re.sub("[0-9]+","#",t)   
    return t

def clean_text_stemmed(t):
    """Accepts a Document 
    """
    t = t.lower()
    # Remove single characters
    t = re.sub("[^A-Za-z0-9]"," ",t)
    # Replace all numbers by a single char
    t = re.sub("[0-9]+","#",t)
    stemmer = snowballstemmer.stemmer('english')
    tfinal = " ".join(stemmer.stemWords(t.split()))    
    return t

# -


