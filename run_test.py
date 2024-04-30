import numpy as np
import pandas as pd
import sys
from utils import load_data, compl_svd_projector, generate_pairs, create_clusters, get_cluster_n
from explore import proximal_gd_sigmoid 
from utils_test import run_method_test
from random import sample

data_path = './sentiment_data/'
# mukherjee et al's sentiment_glove.42B.300d only contains the glove.42B.300d words which appear in Bing Liu's sentiment dataset
# embeddings_path = './sentiment_data/sentiment_glove.42B.300d.txt'
# we use the whole dataset from https://nlp.stanford.edu/projects/glove/
# takes a while to load and reuqires ~5GB of RAM
embeddings_path = './sentiment_data/glove.42B.300d.txt'

# all_names_embed is an np array with shape (num_names, 300)
# 1226 names actually appear in the embeddings from Mukherjee et al's dataset (5897 in names_from_df, so includes duplicates)
embeddings, X, y, vocab, all_names_embed, names_from_df = load_data(data_path, embeddings_path, state=None, names_path=data_path)

"""
#preprocessing to remove duplicates from names
from collections import defaultdict
uniques = defaultdict(int)
for idx,name in enumerate(names_from_df):
    uniques[name] = idx

unique_indices = sorted(list(uniques.values()))
unique_names = [names_from_df[i] for i in unique_indices]
unique_names_embed = all_names_embed[unique_indices]

all_names_embed = unique_names_embed
names_from_df = unique_names
"""

#for i in range(100):
f = open("food/all",'w')
#f = open("random_clusters/random_%d" % i, 'w')
#f = open("clusters_75/run%d" % i, 'w')


"""
# generate random samples of 1226 words to use instead of names to establish baseline results
# we will sample 1226 random words from the ~2M in glove to declare "comparable"
random_indices = sorted(sample(range(len(embeddings)), 1226))
all_names_embed = embeddings.iloc[random_indices].to_numpy()
names_from_df = [embeddings.index[i] for i in random_indices]
"""

"""
# glove.42B.300d is too large to cluster reasonably on my personal computer...
# so intead, sample 10000 random word embeddings and cluster them into 75 clusters
random_indices = sorted(sample(range(len(embeddings)), 10000))
embeddings_sub = embeddings.iloc[random_indices]
num_clusters = 75
clusters = create_clusters(embeddings_sub, num_clusters=num_clusters)
sizes = [len(get_cluster_n(embeddings_sub,clusters,i)[0]) for i in range(75)]
big_enough = [i for i in range(75) if sizes[i] > 100]
i,j = sample(big_enough,2)
i_embed, i_from_df = get_cluster_n(embeddings_sub, clusters, i)
j_embed, j_from_df = get_cluster_n(embeddings_sub, clusters, j)
all_names_embed = np.vstack((i_embed, j_embed))
names_from_df = i_from_df + j_from_df
#g = open("clusters_75/words%d" % i, 'w')
for name in names_from_df:
    g.write(name)
    g.write('\n')
g.close()
print(len(names_from_df))
"""

svd_dims = [3, 10, 50]
proj_face = []
for subspace_d in svd_dims:
    proj = compl_svd_projector(all_names_embed, svd=subspace_d)
    proj_face.append(proj)

################ Get EXPLORE metric ################
np.random.seed(1)

n_pairs_comp = 50000
unique_names_idx = np.unique(names_from_df, return_index=True)[1]
pairs_idx = generate_pairs(len(unique_names_idx), len(unique_names_idx), n_pairs=n_pairs_comp)
comparable_pairs = all_names_embed[unique_names_idx[pairs_idx[0]]] - all_names_embed[unique_names_idx[pairs_idx[1]]]

# In-comparable
n_pairs_incomp = 50000
pos_idx = np.where(y==1)[0]
neg_idx = np.where(y==-1)[0]
pairs_idx = generate_pairs(len(pos_idx), len(neg_idx), n_pairs=n_pairs_incomp)
incomp_pairs = X[pos_idx[pairs_idx[0]]] - X[neg_idx[pairs_idx[1]]]

# Pairs data
X_pairs = np.vstack((comparable_pairs, incomp_pairs))
Y_pairs = np.zeros(n_pairs_comp + n_pairs_incomp)
Y_pairs[:n_pairs_comp] = 1

# Run EXPLORE
Sigma_explore = proximal_gd_sigmoid(X_pairs, Y_pairs, mbs=10000, maxiter=1000)

################ Get name means projector ################
names_proj = np.load('./embeddings/mean_names.npy')
mean_direction = names_proj[:10].mean(axis=0) - names_proj[10:].mean(axis=0)
mean_direction = (mean_direction/np.linalg.norm(mean_direction)).reshape(-1,1)
mean_proj = np.eye(mean_direction.shape[0]) - np.matmul(mean_direction, mean_direction.T)

# don't need to run baseline, bolukbasi, and means every time
# methods = {'baseline': [None, None], 'bolukbasi':[None, None], 'Means': [None, mean_proj],
methods = {'explore':[Sigma_explore, None],
           'FACE-3': [None, proj_face[0]], 'FACE-10': [None, proj_face[1]], 'FACE-50': [None, proj_face[2]]}

# run all WEATs
test1 = ['flowers', 'insects', 'pleasant', 'unpleasant']
test2 = ['instruments', 'weapons', 'pleasant', 'unpleasant']
test3 = ['mental', 'physical', 'temporal', 'permanent']
test4 = ['white', 'black', 'pleasant', 'unpleasant']
test5 = ['white7', 'black7', 'pleasant', 'unpleasant']
test6 = ['white7', 'black7', 'pleasant9', 'unpleasant9']
test7 = ['male_names', 'female_names', 'career', 'family']
test8 = ['math', 'arts', 'male_terms', 'female_terms']
test9 = ['science10', 'arts10', 'male_terms10', 'female_terms10']
test10 = ['young', 'old', 'pleasant9', 'unpleasant9']

all_tests = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10]
result = np.zeros((len(all_tests), len(methods)*2))
for idx, t in enumerate(all_tests):
    X_name, Y_name, A_name, B_name = t
    f.write('Association test %d: target words are %s and %s; attribute words are %s and %s\n' % (idx, X_name, Y_name, A_name, B_name))
    m_i = 0
    for method, (Sigma, proj) in methods.items():
        p_val, z_val = run_method_test(X_name, Y_name, A_name, B_name, f, method=method, Sigma=Sigma, proj=proj)
        result[idx,m_i] = p_val
        result[idx,m_i + 1] = z_val
        m_i += 2
    f.write('\n')

test_idx = np.array([0,3,6,8,9])
test_names = ['FLvINS-PLvUPL', 'EAvAA-PLvUPL', 'MNvFN-CARvFAM', 'SCvART-MTvFT', 'YNGvOLD-PLvUPL']
method_idx = [0,1]#,3,4]
method_idx = np.array([m*2 + 1 for m in method_idx])
#method_names = ['Euclidean', 'Bolukbasi+',
method_names = ['EXPLORE', 'FACE-3']

df = {'test':test_names}
for m_idx, m_name in zip(method_idx, method_names):
    df[m_name] = result[test_idx][:,m_idx]
df = pd.DataFrame(df)
df = pd.melt(df, id_vars="test", var_name="Method", value_name="effect size")
f.write(df.to_string())
f.close()
