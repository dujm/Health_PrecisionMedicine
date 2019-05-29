
# Load packages
# System packages
import os
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


# Word cloud visualization libraries
from scipy.misc import imresize
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter


# Dimensionaly reduction libraries
from sklearn.decomposition import PCA

# Clustering library
from sklearn.cluster import KMeans


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


# Bar Plot: Count by colname
def col_count_plot(df,colname,save_plot_dir):
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
def frequency_plot(df,colname,save_plot_dir):
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
def word_freq_plot(class_corpus, save_plot_dir):
    whole_text_freq = class_corpus.sum()
    fig, ax = plt.subplots()
    label, repetition = zip(*whole_text_freq.most_common(25))
    ax.barh(range(len(label)), repetition, align='center')
    ax.set_yticks(np.arange(len(label)))
    ax.set_yticklabels(label)
    ax.invert_yaxis()
    ax.set_title('Word Distribution Over Whole Text')
    ax.set_xlabel('# of repetitions')
    ax.set_ylabel('Word')
    plt.tight_layout()
    plt.show()
    #save plot
    name= 'word_freq_in_corpus'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    plt.savefig(figname, figdpi = 300)

# Build a word cloud without mask image
def word_cloud_plot_no_mask(corpus,save_plot_dir):
    whole_text_freq = class_corpus.sum()
    wc = WordCloud(max_font_size=300,min_font_size=30,
               max_words=1000,
               width=mask_image.shape[1],
               height=mask_image.shape[0],
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
    whole_text_freq = class_corpus.sum()
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
    name= 'word_cloud_plot'
    figname = '{}{:%Y%m%dT%H%M}.png'.format(os.path.join(save_plot_dir,name), datetime.datetime.now())
    plt.savefig(figname, figdpi = 600)


# PCA plot
def pca_plot(classes, vecs,save_plot_dir):
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
def kmeans_plot(classes, vecs,save_plot_dir):
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
