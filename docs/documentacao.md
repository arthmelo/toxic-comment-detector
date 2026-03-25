# Documentação do Projeto

## Importação do dataset
- Nossos dados estão em um arquivo chamado comentarios_toxicos_ptBR.csv na pasta data/
- O dataset está um conjunto grande de frases de redes sociais classificadas entre 0 e 1, onde 1 é um comentário tóxico.
- O dataset possui uma coluna com os textos normalizados, mas não iremos usar essa coluna.

## Bibliotecas
- Pandas
- Spacy
- re
- scikit-learn

## Primeiros Passos:
- Primeiro importamos nosso dataset;
- Retiramos a coluna com os textos normalizados
- Havia apenas uma linha com dados nulos, então podemos retirar
- Verificamos a quantidade de comentários tóxicos e vimos que possui pouco menos de 3 mil comentários de diferença na divisão.

## Tokenização
Para fazer o processamento TF-IDF e também retirar as stopwords, precisamos tokenizar o texto. Tokenizar é dividir o nossos texto em parte menores, chamadas tokens, que serão nossas palavras.

https://spacy.io/api/tokenizer/
https://www.datacamp.com/pt/blog/what-is-tokenization

## Remoção de Stopwords

As Stopwords são as palavras comuns do texto que não possuem significado relevante para a análise, então queremos tirar palavras como "o", "a", "de", "é", etc.
Além disso queremos retirar sinais de pontuação e acentos, já que não agrega em nada o significado da frase.

https://www.geeksforgeeks.org/nlp/removing-stop-words-nltk-python/


## Dividir entre dados de treino e dados de teste

Para treinar nosso modelo precisaremos dividir o dataset em duas partes. A primeira e maior parte deve ser a de treino, que será os dados que usaremos para treinar nosso modelo. A segunda e menor será o nosso teste, que será para verificar se o nosso modelo realmente aprendeu. A divisão será feita 80% para os dados de treino e 20% para os dados de teste. Nesse caso usaremos a biblioteca scikit-learn train_test_split. Além disso, como citado anteriormente, como há mais casos "não tóxicos", vamos aplicar undersampling para equalizar a quantidade de comentários tóxicos (1) e não tóxicos (0) nos dados de treino, evitando que o modelo aprenda a favorecer a classe majoritária.

https://docs.kanaries.net/pt/topics/Scikit-Learn/sklearn-train-test-split

## TF-IDF

Agora queremos mandar esses dados para o nosso modelo. O problema está que o modelo de Regressão Logística que iremos utilizar não aceita strings, apenas números. Com isso, vamos utilizar a métrica TF-IDF (Term Frequency - Inverse Document Frequency), que além de fazer a conversão, calcula um "peso" para cada palavra. O TF-IDF será **fitado apenas nos dados de treino** (`.fit_transform()`), para evitar vazamento de dados (data leakage). Em seguida, será **aplicado nos dados de teste** (`.transform()`), utilizando o vocabulário aprendido no treino.

## Features e Target
Nossas Features serão o nosso texto tokenizado e sem as stopwords e nosso Target será a coluna toxic (binária).

## Modelo
Usaremos Regressão Logística para o nosso projeto por ser um dos melhores modelos para classificação binária.

## Resultado
Obtivemos uma acurácia de 77% e f1-score de 81% para o "não tóxico" e 73% para os "tóxicos". Isso nos revela que o modelo performa melhor na identificação de comentários não tóxicos do que tóxicos. Isso pode ser explicado pelo fato de que, mesmo após o balanceamento nos dados de treino, a linguagem tóxica tende a ser mais variada e subjetiva, o que dificulta a generalização do modelo. De forma geral, os resultados são satisfatórios para uma primeira versão, considerando a complexidade da tarefa de detecção de toxicidade em português.