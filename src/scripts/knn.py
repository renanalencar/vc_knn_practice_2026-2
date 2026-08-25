# %%
# This mounts your Google Drive to the Colab VM.
from google.colab import drive
drive.mount('/content/drive')

# TODO: Enter the foldername in your Drive where you have saved the unzipped
# assignment folder, e.g. 'cs231n/assignments/assignment1/'
FOLDERNAME = None
assert FOLDERNAME is not None, "[!] Enter the foldername."

# Now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# This downloads the CIFAR-10 dataset to your Drive
# if it doesn't already exist.
%cd /content/drive/My\ Drive/$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /content/drive/My\ Drive/$FOLDERNAME

# %%
# This downloads the CIFAR-10 dataset to your local folder
# if it doesn't already exist.
%cd ../data/cs231n/datasets/
!"C:\Program Files\Git\bin\bash.exe" get_datasets.sh
%cd ../../../notebooks

# %% [markdown]
# # k-Nearest Neighbor (kNN) exercise
# 
# *Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*
# 
# The kNN classifier consists of two stages:
# 
# - During training, the classifier takes the training data and simply remembers it
# - During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples
# - The value of k is cross-validated
# 
# In this exercise you will implement these steps and understand the basic Image Classification pipeline, cross-validation, and gain proficiency in writing efficient, vectorized code.

# %%
# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

# %%
# Load the raw CIFAR-10 data.
cifar10_dir = '../data/cs231n/datasets/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# %%
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# %%
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# %%
from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# %% [markdown]
# We would now like to classify the test data with the kNN classifier. Recall that we can break down this process into two steps: 
# 
# 1. First we must compute the distances between all test examples and all train examples. 
# 2. Given these distances, for each test example we find the k nearest examples and have them vote for the label
# 
# Lets begin with computing the distance matrix between all training and test examples. For example, if there are **Ntr** training examples and **Nte** test examples, this stage should result in a **Nte x Ntr** matrix where each element (i,j) is the distance between the i-th test and j-th train example.
# 
# **Note: For the three distance computations that we require you to implement in this notebook, you may not use the np.linalg.norm() function that numpy provides.**
# 
# First, open `cs231n/classifiers/k_nearest_neighbor.py` and implement the function `compute_distances_two_loops` that uses a (very inefficient) double loop over all pairs of (test, train) examples and computes the distance matrix one element at a time.

# %%
# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# %%
# We can visualize the distance matrix: each row is a single test example and
# its distances to training examples
plt.imshow(dists, interpolation='none')
plt.show()

# %% [markdown]
# **Inline Question 1** 
# 
# Notice the structured patterns in the distance matrix, where some rows or columns are visibly brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)
# 
# - What in the data is the cause behind the distinctly bright rows?
# - What causes the columns?
# 
# $\color{blue}{\textit Your Answer:}$
# 
# Quando nos deparamos com uma imagem de teste que se destaca das demais, geralmente é porque ela é muito diferente da maioria das imagens usadas para treinamento. Isso pode acontecer quando a imagem de teste tem uma cor de fundo estranha, iluminação incomum ou mostra um objeto de uma forma que simplesmente não vêemos nos dados de treinamento. Por outro lado, se vêemos uma imagem de treinamento que é muito diferente da maioria das imagens de teste, isso é um sinal de que essa imagem de treinamento também é um outlier. Ela não se parece com os outros dados que estamos tentando classificar. Elas ajudam a encontrar os pontos de dados nos conjuntos de teste ou treinamento que simplesmente não se encaixam no restante. Por exemplo, imaginemos que estamos tentando reconhecer fotos de cachorros, mas uma das as imagens de teste é a foto de um gato. Essa foto do gato será muito diferente de todas as fotos de cachorros, então ela se destacará. Ou, se tivermos uma imagem de treinamento que seja a foto de um cachorro, mas tirada de um ângulo muito estranho que não vêemos em nenhuma das imagens de teste, isso também vai se destacar. Esses valores discrepantes podem ser importantes de observar porque podem afetar o desempenho do sistema. Se tivermos muitos valores discrepantes nos dados de treinamento, isso pode prejudicar o reconhecimento do sistema. Mas se conseguirmos encontrar esses valores discrepantes e lidar com eles, poderá melhorar o sistema. Portanto, é sempre uma boa ideia analisar atentamente qualquer imagem que pareça muito diferente das outras.
# 
# 

# %%
# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# %% [markdown]
# You should expect to see approximately `27%` accuracy. Now lets try out a larger `k`, say `k = 5`:

# %%
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# %% [markdown]
# You should expect to see a slightly better performance than with `k = 1`.

# %% [markdown]
# **Inline Question 2**
# 
# We can also use other distance metrics such as L1 distance.
# For pixel values $p_{ij}^{(k)}$ at location $(i,j)$ of some image $I_k$, 
# 
# the mean $\mu$ across all pixels over all images is $$\mu=\frac{1}{nhw}\sum_{k=1}^n\sum_{i=1}^{h}\sum_{j=1}^{w}p_{ij}^{(k)}$$
# And the pixel-wise mean $\mu_{ij}$ across all images is 
# $$\mu_{ij}=\frac{1}{n}\sum_{k=1}^np_{ij}^{(k)}.$$
# The general standard deviation $\sigma$ and pixel-wise standard deviation $\sigma_{ij}$ is defined similarly.
# 
# Which of the following preprocessing steps will not change the performance of a Nearest Neighbor classifier that uses L1 distance? Select all that apply. To clarify, both training and test examples are preprocessed in the same way.
# 
# 1. Subtracting the mean $\mu$ ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu$.)
# 2. Subtracting the per pixel mean $\mu_{ij}$  ($\tilde{p}_{ij}^{(k)}=p_{ij}^{(k)}-\mu_{ij}$.)
# 3. Subtracting the mean $\mu$ and dividing by the standard deviation $\sigma$.
# 4. Subtracting the pixel-wise mean $\mu_{ij}$ and dividing by the pixel-wise standard deviation $\sigma_{ij}$.
# 5. Rotating the coordinate axes of the data, which means rotating all the images by the same angle. Empty regions in the image caused by rotation are padded with a same pixel value and no interpolation is performed.
# 
# $\color{blue}{\textit Your Answer:}$ 1, 2 e 3.
# 
# 
# $\color{blue}{\textit Your Explanation:}$ O desempenho de um classificador kNN é basicamente determinado pela ordem das distâncias entre as imagens de teste e as imagens de treinamento. Portanto, se realizarmos qualquer pré-processamento que mantenha essa ordem, isso não afetará o desempenho do classificador. Isso significa que, contanto que as distâncias relativas sejam preservadas, o desempenho do classificador permanecerá o mesmo.
# 
# - 1. **Subtraindo a média $\mu$**: Os novos valores de pixel se tornam $\tilde{p}{ij} = p{ij} - \mu$. A distância L1 entre duas imagens $x$ e $y$ é calculada como $\sum_{i,j} |(x_{ij} - \mu) - (y_{ij} - \mu)| = \sum_{i,j} |x_{ij} - y_{ij}|$. A constante $\mu$ se cancela perfeitamente, deixando a distância idêntica à distância L1 original. O desempenho não será alterado.
# 
# - 2. **Subtraindo a média por pixel $\mu_{ij}$**: Os novos valores de pixel tornam-se $\tilde{p}{ij} = p{ij} - \mu_{ij}$. A distância L1 torna-se $\sum_{i,j} |(x_{ij} - \mu_{ij}) - (y_{ij} - \mu_{ij})| = \sum_{i,j} |x_{ij} - y_{ij}|$. Assim como acima, a média por pixel é cancelada. O desempenho não será alterado.
# 
# - 3. **Subtraindo a média $\mu$ e dividindo pelo desvio padrão $\sigma$**: Quando subtraímos o valor médio e divide pelo desvio padrão, a distância entre os elementos muda de forma simples. Isso é chamado de normalização dos dados. A distância L1 torna-se uma versão escalonada da distância original. É como multiplicar cada distância pelo mesmo número, que é o inverso do desvio padrão. Essa escala não altera quais coisas estão mais próximas umas das outras. Se uma coisa estava mais perto de outra antes, ela ainda estará mais perto depois da escala. A ordem das distâncias permanece a mesma. Portanto, quando estamos procurando os vizinhos mais próximos, ainda encontrará os mesmos. E como as distâncias são apenas escaladas, o desempenho do sistema também não mudará.
# 
# - 4. **Subtraindo a média por pixel $\mu_{ij}$ e dividindo pelo desvio padrão por pixel $\sigma_{ij}$**: Quando subtraímos o valor médio do pixel e divide pelo desvio padrão para cada pixel, estamos essencialmente usando um fator de escala diferente para cada ponto. Isso muda a forma como medimos a distância entre as imagens. Em vez de apenas somar as diferenças, agora estamos dando mais peso a alguns pixels e menos a outros. A distância entre duas imagens agora é uma soma ponderada das diferenças em cada pixel, onde o peso é o inverso do desvio padrão nesse pixel. Isso pode distorcer a percepção da distância entre as imagens, de modo que o que parece ser a correspondência mais próxima antes pode não ser a mesma depois. E isso pode afetar o desempenho do sistema.
# 
# - 5. **Rotação dos eixos de coordenadas dos dados**: A distância L1 depende muito da escolha dos eixos de coordenadas (ela não é invariante à rotação como a distância euclidiana L2). Além disso, rotacionar uma grade de imagem 2D discreta altera o mapeamento espacial dos pixels. Como nenhuma interpolação é realizada, alguns dados de pixel serão deslocados, duplicados ou descartados fora dos limites, o que altera fundamentalmente o cálculo da distância entre as imagens. O desempenho será afetado.
# 

# %%
# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('One loop difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# %%
# Now implement the fully vectorized version inside compute_distances_no_loops
# and run the code
dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('No loop difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# %%
# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# You should see significantly faster performance with the fully vectorized implementation!

# NOTE: depending on what machine you're using, 
# you might not see a speedup when you go from two loops to one loop, 
# and might even see a slow-down.

# %% [markdown]
# ### Cross-validation
# 
# We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily. We will now determine the best value of this hyperparameter with cross-validation.

# %%
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for k in k_choices:
    # Initialize the list for this k
    k_to_accuracies[k] = []
    
    for i in range(num_folds):
        # 1. Isolate the validation set
        X_val = X_train_folds[i]
        y_val = y_train_folds[i]
        
        # 2. Combine all other folds into the training set
        X_train_cv = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        y_train_cv = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])
        
        # 3. Train the classifier with the CV training data
        classifier.train(X_train_cv, y_train_cv)
        
        # 4. Predict labels for the validation set (using the fast 0-loop version)
        y_pred = classifier.predict(X_val, k=k, num_loops=0)
        
        # 5. Compute and store the accuracy
        accuracy = np.mean(y_pred == y_val)
        k_to_accuracies[k].append(accuracy)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

# %%
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# %%
# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.
best_k = 1

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# %% [markdown]
# **Inline Question 3**
# 
# Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.
# 1. The decision boundary of the k-NN classifier is linear.
# 2. The training error of a 1-NN will always be lower than or equal to that of 5-NN.
# 3. The test error of a 1-NN will always be lower than that of a 5-NN.
# 4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
# 5. None of the above.
# 
# $\color{blue}{\textit Your Answer:}$ 2 e 4.
# 
# 
# $\color{blue}{\textit Your Explanation:}$
# - 1. **[Falso]** A fronteira de decisão de um classificador $k$-NN não é linear. Ela é composta por diversas fronteiras complexas que se assemelham a um diagrama de Voronoi (no caso do 1-NN) e se adaptam localmente aos dados de treinamento.
# 
# - 2. **[Verdadeiro]** Ao analisar o modelo 1-NN, é importante observar que o vizinho mais próximo de qualquer ponto de treinamento é, na verdade, o próprio ponto. Isso significa que a distância é essencialmente zero. Portanto, assumindo que não existam pontos idênticos que pertençam a classes diferentes, o modelo não cometerá erros no conjunto de treinamento, ou seja, o erro de treinamento é zero. Por outro lado, se estivermos usando o modelo 5-NN, as coisas podem ficar um pouco mais complexas. Mesmo com pontos de treinamento, o modelo ainda pode errar se a maioria dos 5 vizinhos mais próximos pertencer a uma classe diferente. Isso pode levar a uma taxa de erro maior que zero. Resumindo, o erro de treinamento para o 1-NN será sempre menor ou igual a zero, porque basicamente compara cada ponto consigo mesmo. Mas para o 5-NN, o erro de treinamento pode ser maior que zero, porque considera o voto majoritário dos 5 vizinhos mais próximos, e isso às vezes pode levar a classificações incorretas, mesmo para os dados de treinamento.
# 
# - 3. **[Falso]** O 1-NN é altamente sensível a ruídos e anomalias nos dados de treinamento, o que frequentemente causa sobreajuste. O 5-NN, ao considerar a votação de mais vizinhos, suaviza a fronteira de decisão e normalmente generaliza melhor para dados não vistos, resultando em um erro de teste *menor* do que o 1-NN em cenários do mundo real.
# 
# - 4. **[Verdadeiro]** No algoritmo $k$-NN clássico, a fase de teste (previsão) requer o cálculo da distância entre a nova amostra e **todos** os $N$ pontos presentes no conjunto de treinamento. Assim, a complexidade computacional para classificar cada novo exemplo é $O(N \cdot D)$, crescendo linearmente com o tamanho do conjunto de treinamento.
# 
# 


