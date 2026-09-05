# Detector de Toxicidade em Redes Sociais: Filtragem de Comentários com PLN e Deep Learning

## Sobre o Projeto

Este projeto tem como objetivo criar um modelo de Processamento de Linguagem Natural (PLN) para filtrar comentários em redes sociais, classificando-os como ofensivos ou não.

O *cyberbullying* é um problema frequente nas redes sociais devido ao anonimato e à falta de fiscalização. Assim como a plataforma de jogos Roblox, que recentemente lançou uma inteligência artificial para censurar e reformular palavrões nos chats, este projeto busca promover um ambiente digital mais seguro. 

Com uma filtragem eficiente, realizada antes mesmo da exibição do comentário, é possível proteger os usuários da exposição a mensagens de ódio e intimidação. Para demonstrar a aplicação prática do modelo, foi desenvolvida uma interface que simula o feed de uma rede social fictícia, o **"InstaCão"**.

## Objetivos

O sistema recebe comentários em formato de texto e realiza a classificação binária ("Não Tóxico" ou "Tóxico"). A avaliação do desempenho é feita por meio das métricas de f1-score, recall e matriz de confusão.

## Tecnologias Utilizadas

- Pandas e Numpy: Manipulação dos dados.
- SpaCy e re: Processamento e limpeza de texto em português.
- Scikit-learn: Vetorização (TF-IDF) e treinamento dos modelos clássicos.
- TensorFlow / Keras: Construção e treinamento da Rede Neural Recorrente com células LSTM (RNN).
- Streamlit: Criação da interface gráfica interativa.

## Dataset

O conjunto de dados utilizado foi o [Comentários Tóxicos ptBR no Kaggle](https://www.kaggle.com/datasets/gedorneto/comentrios-toxicos-ptbr).

## Como utilizar

1. Clonar o repositório e entrar na pasta do projeto:

```bash
git clone https://github.com/seu-usuario/toxic-comment-detector.git
cd toxic-comment-detector
```

2. Criar e ativar o ambiente virtual:

```bash
python -m venv .venv
```

- **Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Linux/macOS:**

```bash
source .venv/bin/activate
```

3. Instalar as dependências e o modelo de língua portuguesa do SpaCy:

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

4. (Opcional) Treinar os modelos — baixa o dataset e gera os arquivos `.pkl` e `.keras` na pasta `models/`:

```bash
python treinar_modelos.py
```

> Se não quiser treinar, os modelos já podem estar disponíveis na pasta `models/`.

5. Iniciar a interface do Streamlit:

```bash
streamlit run app.py
```

A aplicação abrirá no navegador, simulando o feed do "InstaCão". Escolha o filtro de toxicidade desejado e digite comentários para testar a censura em tempo real.

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
   - Rede Neural Recorrente com células LSTM bidirecionais (RNN)

## Resultados e Aplicação Prática

A evolução do projeto consistiu em expandir a modelagem original para um ambiente de múltiplos algoritmos. Os modelos baseados no Scikit-learn estabeleceram uma boa fundação de acurácia, enquanto a arquitetura de Rede Neural Recorrente (LSTM) permitiu capturar o contexto sequencial do texto de maneira mais complexa.

Abaixo, os resultados obtidos em cada um dos algoritmos durante a avaliação final no conjunto de testes:

| Modelo | Acurácia | F1-Score (Tóxico) | F1-Score (Não Tóxico) | Recall (Tóxico) |
|--------|----------|-------------------|-----------------------|-----------------|
| Regressão Logística | 77% | 73% | 80% | 68% |
| Naive Bayes | 73% | 67% | 77% | 62% |
| Random Forest | 79% | 77% | 81% | 79% |
| SVM (LinearSVC) | 76% | 73% | 79% | 71% |
| RNN (LSTM) | 76% | 71% | 80% | 89% |

Analisando os resultados acima, o modelo **RNN (LSTM)** foi o grande destaque para o objetivo do projeto. Como a meta principal do sistema de moderação é garantir um ambiente seguro e filtrar o maior número possível de ofensas, a métrica mais importante a ser avaliada é o **Recall da classe "Tóxico"**, pois buscamos minimizar ao máximo os *falsos negativos* (comentários tóxicos que acabam passando despercebidos pelo filtro). 

Com um Recall de **89%**, a rede neural RNN provou ser significativamente superior aos modelos clássicos em não deixar as mensagens de ódio passarem. Apesar do Random Forest apresentar a melhor acurácia global (79%), o seu recall (79%) indica que ele ainda deixaria passar uma quantidade considerável de comentários maliciosos comparado à RNN.

Para colocar a teoria em prática, o repositório conta com o arquivo app.py. Ao executá-lo, o usuário interage com um simulador de celular via web, onde é possível selecionar qual dos cinco modelos atuará como o filtro de censura em tempo real para os comentários que são digitados na tela.

## Próximos Passos

- Tratar e normalizar gírias ou erros ortográficos intencionais que burlam a identificação atual.
- Explorar arquiteturas baseadas em Transformers, como BERTimbau.

## Referências
https://veja.abril.com.br/tecnologia/roblox-lanca-inteligencia-artificial-que-censura-e-reformula-palavroes-nos-chats/

https://www.geeksforgeeks.org/nlp/rnn-for-text-classifications-in-nlp/

https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

https://slds-lmu.github.io/i2ml/chapters/07_forests/07-00-nutshell-random-forest/

https://www.datacamp.com/pt/tutorial/random-forests-classifier-python

https://slds-lmu.github.io/i2ml/chapters/03_supervised_classification/