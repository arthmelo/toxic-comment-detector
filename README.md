# Detector de Toxicidade em Redes Sociais: Filtragem de Comentários com PLN e Deep Learning

## Sobre o Projeto

Esse projeto é uma solução automatizada baseada em Processamento de Linguagem Natural (PLN) para detectar e filtrar comentários tóxicos e ofensivos em redes sociais. O objetivo principal é mostrar forma de proteger os usuários de discursos de ódio na internet, criando um ambiente digital mais seguro através de filtros de comentários nocivos antes que elas impactem os usuários.

Para demonstrar a utilidade prática do algoritmo, foi construída uma interface interativa simples que simula o feed de uma rede social fictícia batizada de "Instagrão". Nesta aplicação, o usuário pode digitar comentários e selecionar diferentes modelos de Inteligência Artificial para atuar como um filtro de censura em tempo real.

![image](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.rawpixel.com%2Fimage_800%2FcHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTEwL3JtNTg1ZGVzaWduLXMtYS1yZW1peC0wOC5qcGc.jpg&f=1&nofb=1&ipt=4bd80f61826d65a8fac47efeb2a729e4303659fbd67a2c2b99c640d9176ab97f)

## Objetivos

O sistema recebe comentários em formato de texto e realiza a classificação binária ("Não Tóxico" ou "Tóxico"). A avaliação do desempenho é feita por meio das métricas de acurácia, f1-score e matriz de confusão.

## Tecnologias Utilizadas

- Pandas e Numpy: Manipulação dos dados.
- SpaCy e re: Processamento e limpeza de texto em português.
- Scikit-learn: Vetorização (TF-IDF) e treinamento dos modelos clássicos.
- TensorFlow / Keras: Construção e treinamento da Rede Neural Convolucional (CNN).
- Streamlit: Criação da interface gráfica interativa.

## Dataset

O conjunto de dados utilizado foi o [Comentários Tóxicos ptBR no Kaggle](https://www.kaggle.com/datasets/gedorneto/comentrios-toxicos-ptbr).

## Modelos e Abordagem

O fluxo do projeto foi estruturado em um pipeline automatizado por meio do script principal, que executa as seguintes etapas:

1. Pré-processamento: Os textos são convertidos para letras minúsculas. Remove-se a pontuação utilizando expressões regulares e as palavras irrelevantes (stopwords) via spaCy (modelo pt_core_news_sm).
2. Divisão de Dados: O dataset foi dividido em 80% para treino e 20% para teste, mantendo a proporção de classes (stratify) para preservar a distribuição original de comentários normais e tóxicos.
3. Vetorização:
   - Para os modelos clássicos, utilizou-se o TfidfVectorizer limitado a 10.000 features.
   - Para o modelo de Deep Learning, utilizou-se o Tokenizer do Keras padronizando o tamanho das sentenças via pad_sequences.
4. Treinamento e Avaliação:
   Foram treinados e persistidos (em arquivos .pkl e .keras) cinco modelos diferentes para fins de comparação empírica:
   - Regressão Logística
   - Naive Bayes (MultinomialNB)
   - Random Forest
   - Support Vector Machine (LinearSVC)
   - Rede Neural Convolucional 1D (CNN)

## Resultados e Aplicação Prática

A evolução do projeto consistiu em expandir a modelagem original para um ambiente de múltiplos algoritmos. Os modelos baseados no Scikit-learn estabeleceram uma boa fundação de acurácia, enquanto a arquitetura de Rede Neural permitiu capturar contexto sequencial do texto de maneira mais complexa.

Abaixo, os resultados obtidos em cada um dos algoritmos durante a avaliação final no conjunto de testes:

| Modelo | Acurácia | F1-Score (Tóxico) | F1-Score (Não Tóxico) |
|--------|----------|-------------------|-----------------------|
| Regressão Logística | 77% | 73% | 80% |
| Naive Bayes | 73% | 67% | 77% |
| Random Forest | 79% | 77% | 81% |
| SVM (LinearSVC) | 76% | 73% | 79% |
| CNN 1D | 79% | 79% | 79% |

Para colocar a teoria em prática, o repositório conta com o arquivo app.py. Ao executá-lo, o usuário interage com um simulador de celular via web, onde é possível selecionar qual dos cinco modelos atuará como o filtro de censura em tempo real para os comentários que são digitados na tela.

## Próximos Passos

- Otimizar os hiperparâmetros dos modelos via GridSearch.
- Implementar validação cruzada no pipeline principal.
- Tratar e normalizar gírias ou erros ortográficos intencionais que burlam a identificação atual.
- Explorar arquiteturas baseadas em Transformers, como BERTimbau.
