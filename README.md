# Detector de Toxidade em Redes Sociais
## 📌 Sobre o Projeto
- Esse projeto visa criar um modelo de PLN (Processamento de Linguagem Natural) para realizar filtragens em comentários de uma rede social para classificar se eles são ofensivos ou se contém ameaças e intimidações.
  
- O cyberbullying tem se tornado cada vez mais frequentes nas redes sociais hoje em dia. O anônimato e a falta de fiscalização permite com que as pessoas fiquem a vontade para disparar ódio nos comentários. Com uma filtragem eficiente antes mesmo do envio do comentário, jovens e adultos que enfrentam essas situações poderiam ser protegidos desses tipos de comentários.

![image](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimages.rawpixel.com%2Fimage_800%2FcHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTEwL3JtNTg1ZGVzaWduLXMtYS1yZW1peC0wOC5qcGc.jpg&f=1&nofb=1&ipt=4bd80f61826d65a8fac47efeb2a729e4303659fbd67a2c2b99c640d9176ab97f)

## 🎯 Objetivos
- O modelo vai classificar os comentários como "não ofensivos" e "ofensivos" para os que contiverem mensagens com algum tipo de ofensa, ameaça ou intimidação.
- A avaliação será feita pelas métricas de acurácia e f1-score.

## 🛠️ Tecnologias Utilizadas
### Bibliotecas
- Pandas
- Spacy
- re
- scikit-learn

## 📊 Dataset
[Comentários Tóxicos ptBR no Kaggle](https://www.kaggle.com/datasets/gedorneto/comentrios-toxicos-ptbr)

## 🧠 Modelo e Abordagem

- Pré-processamento: os textos foram convertidos para letras minúsculas, com remoção de pontuação, números e acentos via re. Em seguida, foram tokenizados com o tokenizador do spaCy para português (pt_core_news_sm) e tiveram suas stopwords removidas.
- Divisão dos dados: o dataset de 29.921 entradas foi dividido em 80% treino e 20% teste com train_test_split, usando stratify=y para manter a proporção original das classes (16.412 não tóxicos / 13.509 tóxicos) em ambos os conjuntos.
- Vetorização: os textos pré-processados foram convertidos para representação numérica com TfidfVectorizer, limitado às 10.000 features mais relevantes. O fit foi feito exclusivamente nos dados de treino para evitar data leakage.
- Modelo: foi treinada uma Regressão Logística, algoritmo eficiente e interpretável para classificação binária de texto.
- Avaliação: o modelo foi avaliado com classification_report, obtendo 77% de acurácia geral, com f1-score de 81% para comentários não tóxicos e 73% para tóxicos.

## 📈 Resultados

Obtivemos uma acurácia de 77% e f1-score de 81% para o "não tóxico" e 73% para os "tóxicos". Isso nos revela que o modelo performa melhor na identificação de comentários não tóxicos do que tóxicos. Isso pode ser explicado pelo fato de que, mesmo após o balanceamento nos dados de treino, a linguagem tóxica tende a ser mais variada e subjetiva, o que dificulta a generalização do modelo. De forma geral, os resultados são satisfatórios para uma primeira versão, considerando a complexidade da tarefa de detecção de toxicidade em português.

## ⏭️ Próximos Passos

- Tratar o desbalanceamento
- Aumentar dataset
- Explorar outros modelos
- Validação Cruzada
- Tratar gírias
- Implementar parte gráfica
