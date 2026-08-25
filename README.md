# Classificação de Imagens com k-Vizinhos Mais Próximos (k-NN)

Este repositório contém uma tarefa para explorar o algoritmo k-Vizinhos Mais Próximos (k-NN) para classificação de imagens, baseado no material do curso CS231n de Stanford. O objetivo deste exercício é implementar e avaliar um classificador k-NN no conjunto de dados CIFAR-10.

## Estrutura do Projeto

- `notebooks/knn.ipynb`: Notebook Jupyter contendo o exercício principal, visualizações e questões.
- `notebooks/cs231n/`: Módulos Python de suporte contendo a implementação do classificador e utilitários de dados.
- `data/get_datasets.sh` / `data/get_datasets.ps1`: Scripts para baixar o conjunto de dados CIFAR-10.
- `TAREFA.md`: Instruções originais da tarefa.
- `RESOLUCAO.md`: Explicacão das modificações de código e respostas da tarefa.

## Requisitos

- Python >= 3.13
- Um gerenciador de pacotes como `uv` (recomendado) ou `pip`

## Instruções de Configuração

1. **Instalar Dependências** O projeto utiliza o `uv` para gerenciamento de dependências. Você pode instalar os pacotes necessários (como `numpy`, `matplotlib` e `ipykernel`) executando:

```bash
uv sync
```

2. **Baixar o Conjunto de Dados** Antes de executar o notebook, você precisa baixar o conjunto de dados CIFAR-10. Você pode fazer isso diretamente do Jupyter Notebook (`knn.ipynb`) ou manualmente executando o script de download:

- **Windows:** Execute `.\data\get_datasets.ps1` em um console do PowerShell.

- **Linux/macOS:** Execute `bash data/get_datasets.sh`.

3. **Execute o Notebook**
Inicie seu ambiente Jupyter Notebook e abra o arquivo `notebooks/knn.ipynb`:

```bash
uv run jupyter notebook
```

Siga as instruções dentro do notebook para concluir as implementações e responder às perguntas.

## Objetivos da Tarefa

- Compreender o pipeline básico de Classificação de Imagens e validação cruzada.
- Adquirir proficiência na escrita de código Python eficiente e vetorizado usando NumPy.
- Implementar cálculos de matriz de distância (com dois loops, um loop e zero loops).
- Prever rótulos com base nos $k$ vizinhos mais próximos.
- Analisar as diferenças de desempenho entre as implementações.

## Documentos

- [Tarefa](TAREFA.md)
- [Resolução](RESOLUCAO.md)