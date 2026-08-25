# Explicação da Fórmula da Distância Euclidiana (L2)

A fórmula apresentada:
```python
dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
```
calcula a Distância Euclidiana entre um dado ponto do conjunto de teste (amostra $i$) e um ponto do conjunto de treinamento (amostra $j$). A Distância Euclidiana é a medida do segmento de reta entre dois pontos em um espaço euclidiano (n-dimensional).

Vamos detalhar cada parte do código:

1. **`X[i] - self.X_train[j]`**:
   - `X[i]` representa o vetor de características (features) da $i$-ésima amostra do conjunto de teste.
   - `self.X_train[j]` representa o vetor de características da $j$-ésima amostra do conjunto de treinamento.
   - Essa operação realiza a subtração elemento a elemento (element-wise) entre os dois vetores. O resultado é um vetor de diferenças das características entre os dois exemplos.

2. **`**2`**:
   - Eleva ao quadrado cada elemento do vetor resultante da subtração anterior. Isso possui dois efeitos principais: garante que todas as diferenças se tornem números positivos (para que diferenças negativas não anulem as positivas na soma) e penaliza mais fortemente as diferenças grandes.

3. **`np.sum(...)`**:
   - A função `np.sum` soma todos os valores (as diferenças quadradas) do vetor resultante do passo anterior. Matematicamente, corresponde ao somatório $\sum (x_k - y_k)^2$ em todas as dimensões (características).

4. **`np.sqrt(...)`**:
   - Por fim, a função `np.sqrt` calcula a raiz quadrada de toda a soma. Isso conclui a clássica fórmula da Distância Euclidiana: $d(x, y) = \sqrt{\sum_{k=1}^{n} (x_k - y_k)^2}$. 
   - O valor escalar resultante é, então, armazenado na matriz de distâncias `dists` na linha `i` (índice da amostra de teste) e coluna `j` (índice da amostra de treinamento).

---

## Cálculo com Um Loop (Vetorização Parcial)

A fórmula para o cálculo com um loop é:
```python
dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
```

Esta operação calcula a Distância Euclidiana entre o $i$-ésimo ponto de teste e **todos** os pontos de treinamento simultaneamente, aproveitando o recurso de *broadcasting* da biblioteca NumPy. 

Detalhamento da operação:

1. **`X[i] - self.X_train`**:
   - `X[i]` é um vetor 1D representando a $i$-ésima amostra de teste, com formato `(D,)`, onde $D$ é o número de características (features).
   - `self.X_train` é uma matriz 2D contendo todas as amostras de treinamento, com formato `(num_train, D)`.
   - Por meio do *broadcasting*, o NumPy implicitamente "estica" (replica) o vetor `X[i]` para subtraí-lo de cada linha de `self.X_train`. O resultado é uma matriz 2D de formato `(num_train, D)`, contendo as diferenças elemento a elemento entre o ponto de teste $i$ e todos os pontos de treinamento.

2. **`**2`**:
   - Eleva ao quadrado cada elemento da matriz resultante. O formato continua sendo `(num_train, D)`.

3. **`np.sum(..., axis=1)`**:
   - A função `np.sum` realiza o somatório, e o parâmetro `axis=1` indica que a soma deve ser feita ao longo do **eixo 1** (as colunas/características), para cada linha (amostra de treinamento). 
   - O resultado é um vetor 1D de formato `(num_train,)`, onde cada elemento representa a soma das diferenças quadradas entre o ponto de teste $i$ e uma amostra de treinamento.

4. **`np.sqrt(...)`**:
   - Por fim, a raiz quadrada é calculada para cada elemento do vetor 1D.
   - O vetor final de distâncias é atribuído à linha correspondente na matriz de distâncias: `dists[i, :]`. Assim, todas as distâncias para o $i$-ésimo ponto de teste são preenchidas de uma só vez, eliminando a necessidade do loop interno (sobre `j`).

---

## Cálculo Totalmente Vetorizado (Sem Loops)

A fórmula para o cálculo sem nenhum loop é:
```python
dists = np.sqrt(np.sum(X**2, axis=1, keepdims=True)
                + np.sum(self.X_train**2, axis=1)
                - 2 * np.dot(X, self.X_train.T))
```

Esta operação calcula a Distância Euclidiana entre **todos** os pontos de teste e **todos** os pontos de treinamento simultaneamente. Ela se baseia na expansão algébrica do quadrado da diferença (produto notável):

$$(x - y)^2 = x^2 + y^2 - 2xy$$

No contexto de vetores, a distância quadrada entre um vetor de teste $x$ e um vetor de treinamento $y$ pode ser expressa como: $||x||^2 + ||y||^2 - 2 x \cdot y^T$. O código implementa exatamente essa lógica matricial.

Detalhamento da operação:

1. **`np.sum(X**2, axis=1, keepdims=True)`**:
   - `X**2` eleva ao quadrado todos os elementos da matriz de teste `X`.
   - `np.sum(..., axis=1)` soma os quadrados de cada amostra (ao longo das características), calculando o equivalente a $||x||^2$.
   - `keepdims=True` garante que o resultado mantenha as dimensões originais, gerando uma matriz de forma `(num_test, 1)` em vez de um vetor 1D. Isso é crucial para o alinhamento adequado no *broadcasting* da soma subsequente.

2. **`np.sum(self.X_train**2, axis=1)`**:
   - De forma análoga, calcula a soma dos quadrados das características para cada amostra de treinamento, $||y||^2$. 
   - O resultado é um vetor 1D de formato `(num_train,)`.
   - Quando o NumPy soma a matriz `(num_test, 1)` resultante do passo anterior com este vetor `(num_train,)`, ele utiliza *broadcasting* para formar uma nova matriz 2D `(num_test, num_train)`. Esta matriz conterá os valores de $||x||^2 + ||y||^2$ para todas as combinações possíveis entre amostras de teste e treinamento.

3. **`- 2 * np.dot(X, self.X_train.T)`**:
   - `self.X_train.T` é a matriz transposta de `self.X_train`, que possui formato `(D, num_train)`.
   - `np.dot(X, self.X_train.T)` realiza a multiplicação de matrizes entre `X` `(num_test, D)` e a transposta do treino. O resultado é uma matriz `(num_test, num_train)`, onde cada posição contém o produto escalar (dot product) entre uma amostra de teste e uma amostra de treinamento ($x \cdot y^T$).
   - Multiplicamos por `-2` para formar o último termo da expansão: $-2xy$.

4. **Soma Final e `np.sqrt(...)`**:
   - A soma total desses três termos resulta na matriz de distâncias quadradas completa, de tamanho `(num_test, num_train)`.
   - Finalmente, `np.sqrt(...)` calcula a raiz quadrada de cada elemento dessa matriz.
   - O resultado final é a matriz `dists` preenchida de uma vez só, de forma extremamente eficiente e totalmente vetorizada, sem a necessidade de iterar sobre nenhum laço `for` na linguagem Python.

---

## Encontrando a Classe dos Vizinhos (Predição)

Após calcular a matriz de distâncias entre os pontos de teste e de treinamento, o algoritmo precisa encontrar os `k` vizinhos mais próximos para cada ponto de teste e prever a sua classe. O trecho de código abaixo (executado dentro de um laço para cada amostra de teste `i`) realiza essa tarefa:

```python
closest_y = self.y_train[np.argsort(dists[i])[:k]].tolist()

values, counts = np.unique(closest_y, return_counts=True)
y_pred[i] = values[np.argmax(counts)]
```

Detalhamento da operação:

1. **`np.argsort(dists[i])`**:
   - `dists[i]` é um vetor que contém as distâncias da $i$-ésima amostra de teste para todas as amostras de treinamento.
   - `np.argsort` não ordena os valores em si, mas retorna os **índices** que ordenariam esse vetor em ordem crescente. Ou seja, o primeiro elemento do vetor resultante é o índice da amostra de treinamento que tem a menor distância (o vizinho mais próximo), o segundo é o segundo mais próximo, e assim por diante.

2. **`[:k]`**:
   - Aplica um fatiamento (*slicing*) para pegar apenas os `k` primeiros índices do vetor retornado por `np.argsort`. Estes são os índices exatos dos `k` vizinhos mais próximos no conjunto de treino.

3. **`self.y_train[...].tolist()`**:
   - Utiliza esses índices para acessar o array de rótulos originais de treinamento (`self.y_train`). Isso nos entrega os rótulos (classes) reais desses `k` vizinhos. O resultado é transformado numa lista padrão do Python através do método `.tolist()` e armazenado na variável `closest_y`.

4. **Votação Majoritária (`np.unique` e `np.argmax`)**:
   - O algoritmo k-NN faz a predição através de uma votação entre os `k` vizinhos. O rótulo que aparecer mais vezes "ganha".
   - `np.unique(closest_y, return_counts=True)` analisa a lista de rótulos dos vizinhos (`closest_y`) e retorna dois arrays: `values` (as classes únicas que apareceram) e `counts` (a quantidade de votos/vezes que cada classe apareceu).
   - `np.argmax(counts)` encontra o índice onde se localiza o valor máximo no array `counts`, ou seja, identifica qual foi a contagem vencedora.
   - Por fim, `values[np.argmax(counts)]` extrai a classe correspondente a essa contagem vencedora e a salva em `y_pred[i]`. É importante notar que em caso de empates na contagem, o `np.argmax` retorna o primeiro índice encontrado. Como `np.unique` já ordena as classes em `values` de forma crescente, a implementação quebra os empates escolhendo automaticamente o rótulo de menor valor numérico.

---

## Divisão do Dataset em Folds (Treino e Validação)

A primeira etapa preparatória para a validação cruzada consiste em fatiar o conjunto de dados de treinamento em pequenos blocos, chamados de *folds*. O trecho responsável por isso é:

```python
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
```

Detalhamento da operação:

1. **`np.array_split(X_train, num_folds)`**:
   - Esta função do NumPy divide o array original de dados de treinamento (`X_train`) em partes de tamanho igual ou aproximado, de acordo com a quantidade definida na variável `num_folds` (geralmente 5).
   - Diferente da função `np.split`, que exige que a divisão seja exata, o `np.array_split` distribui os elementos de maneira balanceada mesmo se o tamanho total do array não for perfeitamente divisível por `num_folds`. O resultado é uma lista contendo as `num_folds` fatias (arrays menores).

2. **`np.array_split(y_train, num_folds)`**:
   - Executa exatamente o mesmo processo para o array de rótulos (`y_train`).
   - É fundamental que os dados de entrada (`X`) e seus rótulos correspondentes (`y`) sejam divididos de forma idêntica. Assim, a $i$-ésima fatia de características sempre corresponderá à $i$-ésima fatia de rótulos, mantendo o alinhamento das amostras.

---

## Validação Cruzada (Cross-Validation) para a Escolha de k

O trecho de código a seguir implementa a validação cruzada k-fold (neste caso, `num_folds = 5`) para avaliar a precisão do classificador k-Nearest Neighbor para diferentes valores de `k` e, assim, ajudar a escolher o hiperparâmetro ideal.

```python
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
```

Detalhamento da operação:

1. **Laços de Repetição (`for k in k_choices` e `for i in range(num_folds)`)**:
   - O laço externo itera sobre cada valor possível de `k` (a quantidade de vizinhos) que desejamos testar. Para cada `k`, criamos uma lista vazia em `k_to_accuracies[k]` para armazenar as precisões calculadas em cada *fold* (fatia).
   - O laço interno itera `num_folds` vezes (normalmente 5). Em cada iteração `i`, uma parte diferente dos dados de treinamento é separada para ser usada como conjunto de validação.

2. **Isolar o Conjunto de Validação (`X_val`, `y_val`)**:
   - `X_train_folds[i]` seleciona a $i$-ésima fatia dos dados para ser o conjunto de validação, usado exclusivamente para testar as previsões do modelo em dados "não vistos" durante esta rodada.
   
3. **Combinar os Folds Restantes para Treino (`X_train_cv`, `y_train_cv`)**:
   - A expressão `X_train_folds[:i] + X_train_folds[i+1:]` cria uma lista com todas as fatias de treinamento **exceto** a $i$-ésima fatia.
   - O `np.concatenate` une todas essas fatias restantes em um único array NumPy contínuo. Este passa a ser o conjunto de dados de treinamento da validação cruzada para esta iteração.

4. **Treinar o Classificador**:
   - `classifier.train(X_train_cv, y_train_cv)` fornece os dados de treinamento combinados para o modelo. No caso do k-NN, o "treinamento" consiste unicamente em memorizar e armazenar os dados e seus rótulos na memória (*lazy learning*).

5. **Prever Rótulos e Calcular Precisão**:
   - `classifier.predict(X_val, k=k, num_loops=0)` prevê os rótulos do conjunto de validação `X_val`, utilizando a implementação otimizada e sem loops (`num_loops=0`) para descobrir e avaliar os `k` vizinhos mais próximos.
   - A precisão (*accuracy*) é extraída através de `np.mean(y_pred == y_val)`, isto é, o cálculo da proporção (média) de acertos entre os rótulos previstos e os rótulos reais. Esse valor é salvo na respectiva lista em `k_to_accuracies[k]`, de modo que no final da execução, cada valor de `k` tenha `num_folds` métricas de precisão.
