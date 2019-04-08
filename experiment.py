# coding: utf-8
import argparse

parser = argparse.ArgumentParser(description='Do document clustering experiment.')
parser.add_argument("--raw",      action='store_true')
parser.add_argument("--title",    action='store_true')
parser.add_argument("--intro",    action='store_true')
parser.add_argument("--stopword", action='store_true')
parser.add_argument("--stemer",   action='store_true')
parser.add_argument("--lemstop",  action='store_true')
parser.add_argument("--lemmer",   action='store_true')

parser.add_argument("--ngram",  type=int,   choices=range(1, 11),)
parser.add_argument("--min_df", type=int,   choices=range(1, 11),)
parser.add_argument("--max_df", type=float, ) #0.05, 0.1 ... 0.95
parser.add_argument("--dr",     type=int,   choices=range(1, 101),)

parser.add_argument('--methods', nargs='+', choices='km em ac aa aw db'.split(),
                    default='km em ac aa aw db'.split(),)

args = parser.parse_args() 
#args = parser.parse_args('--dr 10'.split()) # For testing
print(">" + str(args))



import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix

pre_received, lemma_received = (False, False)
data, all_tokens, morfs = ([],[],[])
def get_pre_data():
    global pre_received, data, all_tokens
    if not pre_received:
        with open("delfi_pre.json", "r") as read_file:
            data = json.load(read_file)
        all_tokens = [" ".join(d["tokens"] + d["stop_tokens"]) for d in data]
        pre_received = True
                        
def get_lemma_data():
    global lemma_received, morfs
    if not lemma_received:
        with open("delfi_lemmas.json", "r") as read_file:
            lemmaData = json.load(read_file)
        morfs = [re.findall("<word=\"(.*)\" lemma=\"(\w*).*\" type=\"(.*)\"", d["lemms"]) for d in lemmaData]
        lemma_received = True

def make_dataset(experiment_data, **kwargs):
    vectorizer = TfidfVectorizer(**kwargs)
    matrix = vectorizer.fit_transform(experiment_data)
    #print(matrix.shape)
  #  quit()
    names = vectorizer.get_feature_names()
    print('>' + str(vectorizer))
    return [{"matrix" : matrix, "names" : names}]  

get_pre_data() # TODO run these only if needed
get_lemma_data()
datasets = []

if args.raw:  
    datasets += make_dataset(all_tokens)
if args.title:
    datasets += make_dataset([re.sub("[\W\d_]+", " ", d["title"]).lower() for d in data])
if args.intro:
    datasets += make_dataset([re.sub("[\W\d_]+", " ", d["intro"]).lower() for d in data])
if args.stopword:
    datasets += make_dataset([" ".join(d["tokens"])                    for d in data])
if args.stemer:
    datasets += make_dataset([" ".join(d["stems"]  + d["stop_stems"])  for d in data])

if args.lemstop:
    datasets += make_dataset([" ".join([l[0] for l in m if l[2].startswith(("dkt", "vksm", "bdv"))]) for m in morfs])
if args.lemmer:
    datasets += make_dataset([" ".join([l[1] for l in m]) for m in morfs])
    
if args.ngram:
    n = args.ngram
    datasets += make_dataset(all_tokens, analyzer = 'char_wb', ngram_range = (n,n))
if args.min_df:
    df = args.min_df
    datasets += make_dataset(all_tokens, min_df = df)
if args.max_df:
    df = args.max_df
    datasets += make_dataset(all_tokens, max_df = df)
if args.dr:
    svd = TruncatedSVD(args.dr)
    lsa = make_pipeline(svd, Normalizer(copy=False)) 
    X = make_dataset(all_tokens)[0]
    matrix = csr_matrix(lsa.fit_transform(X["matrix"]))
    datasets += [{"matrix" : matrix, "names" : X["names"]}]

#print('>' + str(datasets[0]["matrix"]))

category_names = ['Auto', 'Veidai', 'Sportas', 'Mokslas', 'Verslas']
categorys  = np.array([category_names.index(d["categorys"]) for d in data])




from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
K = 5
jobs = -1

methods = []

if "km" in args.methods:
    KMtitle = "K-means"
    KMmodel = KMeans(n_clusters=K,
#                  max_iter=1,
#                  n_init=1,
                 n_jobs=jobs,
                 random_state=42,)
    methods += [{"model": KMmodel, "title": KMtitle}]
    
if "em" in args.methods:
    EMtitle = "Expectation–maximization"
    EMmodel = GaussianMixture(n_components=K,
                              covariance_type='diag',
#                         n_init=10,
                              random_state=42,)
    methods += [{"model": EMmodel, "title": EMtitle}]
    
if "ac" in args.methods:
    ACtitle = "Complete-linkage clustering"
    ACmodel = AgglomerativeClustering(n_clusters=K,
                                      linkage='complete',)
    methods += [{"model": ACmodel, "title": ACtitle}]
    
if "aa" in args.methods:
    AAtitle = "Average-linkage clustering"
    AAmodel = AgglomerativeClustering(n_clusters=K,
                                  linkage='average',)
    methods += [{"model": AAmodel, "title": AAtitle}]
    
if "aw" in args.methods:
    AWtitle = "Ward-linkage clustering"
    AWmodel = AgglomerativeClustering(n_clusters=K,
                                  linkage='ward',)
    methods += [{"model": AWmodel, "title": AWtitle}]
    
if "db" in args.methods:
    DBSCANtitle = "DBSCAN"
    DBSCANmodel = DBSCAN(n_jobs = jobs,)
#    methods += [{"model": DBSCANmodel, "title": DBSCANtitle}]
print(methods)




import itertools
from sklearn.metrics import *
from scipy.stats import mode

def print_top_terms(model, terms):
#    print("Top terms per cluster:")
    centers = model.cluster_centers_ if isinstance(model, KMeans) else model.means_
    order_centroids = centers.argsort()[:, ::-1]
    for i in range(K):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

def get_new_labels(clusters):
    new_labels = np.zeros_like(clusters)
    print("New labels:")
    for i in range(K):
        mask = (clusters == i)
        closest_category = mode(categorys[mask])[0][0]
        new_labels[mask] = closest_category
        print("{} -> {}({})".format(i, closest_category, category_names[closest_category]))
    print(np.bincount(new_labels))
    return new_labels

def print_metrics(y_pred):
    print("Rand        %.3f" %(adjusted_rand_score(categorys, y_pred)))
    print("Homogeneity %.3f" %(homogeneity_score(categorys, y_pred)))
    print("Homogeneity %.3f" %(completeness_score(categorys, y_pred)))

def plot_confusion_matrix(y_pred, title='clusters'):
    cm = confusion_matrix(categorys, y_pred)
    print(cm)




def metrics_and_martix(clusters, m):
    print_metrics(clusters)
    plot_confusion_matrix(clusters, title=m['title'])
    new_labels = get_new_labels(clusters)
    print_metrics(new_labels)
    plot_confusion_matrix(new_labels, title=m['title'])
    
def analyse(m, data):
    model = m['model']
    print('\n' + m['title'] + " results")
    dataset = data["matrix"]
    if m['title'] == KMtitle:
        clusters = model.fit_predict(dataset)
        print(np.unique(clusters, return_counts=True)[1])
        
        print_top_terms(model, data["names"])
        metrics_and_martix(clusters, m)
        
    if m['title'] == EMtitle:
        model.fit(dataset.toarray())
        clusters = model.predict(dataset.toarray())
        print(np.unique(clusters, return_counts=True))
        
        print_top_terms(model, data["names"])
        metrics_and_martix(clusters, m)
        
    if m['title'] in [ACtitle, AAtitle, AWtitle]:
        clusters = model.fit_predict(dataset.toarray())
        print(np.unique(clusters, return_counts=True))
        
        metrics_and_martix(clusters, m)
        
    if m['title'] == DBSCANtitle:
        for e in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            for m in [3, 4, 5, 6, 7, 8]:
                model.set_params(eps = e, min_samples = m,)
                clusters = model.fit_predict(dataset)
                
                results = np.unique(clusters, return_counts=True)
                if results[0][0] == -1: #if there was noise 
                    n_noise    = results[1][0]
                    n_clusters = np.sort(results[1][1:])[::-1]
                else:
                    n_noise    = 0      #if there was no noise 
                    n_clusters = np.sort(results[1])[::-1]
                print ("ε=%.1f min=%i: noise=%4i clusters=%3i top10=%s" 
                       %(e, m, n_noise, len(n_clusters), n_clusters[:10]))
    else:
        print(m)
        
import time

for dataset in datasets:
    print('>' + str(dataset["matrix"].shape))
    for method in methods:
        start_time = time.time()
        analyse(method, dataset)
        print (">", method['title'], time.time() - start_time, "to run")

