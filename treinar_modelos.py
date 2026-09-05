import pandas as pd
import numpy as np
import spacy
import re
import joblib
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.makedirs('./models', exist_ok=True)
nlp = spacy.load("pt_core_news_sm")

def preProcessar(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^a-záéíóúâêôãõüç\s]', ' ', texto)
    doc = nlp(texto)
    return " ".join([token.text for token in doc if not token.is_stop and token.text.strip()])

print("Carregando dados...\n")
df = pd.read_csv('./data/comentarios_toxicos_ptBR.csv')
df.dropna(subset=['text', 'toxic'], inplace=True)

if 'text_norm' in df.columns:
    print("Usando coluna text_norm existente...\n")
    df.dropna(subset=['text_norm'], inplace=True)
    df['text_normalized_str'] = df['text_norm']
else:
    print("Processando textos..\n.")
    df['text_normalized_str'] = df['text'].apply(preProcessar)

X = df['text_normalized_str']
y = df['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
)

tfidf = TfidfVectorizer(max_features=10000)
X_train_vec = tfidf.fit_transform(X_train)
joblib.dump(tfidf, './models/tfidf.pkl')
print("--tfidf.pkl salvo\n")

MAX_WORDS = 20000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_nn)
with open('./models/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("--tokenizer.pkl salvo\n")

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_nn), maxlen=MAX_LEN, padding='post', truncating='post')
X_val_pad = pad_sequences(tokenizer.texts_to_sequences(X_val_nn), maxlen=MAX_LEN, padding='post', truncating='post')

y_train_nn = np.array(y_train_nn)
y_val_nn = np.array(y_val_nn)

print("Treinando Regressão Logística...\n")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_vec, y_train)
joblib.dump(lr, './models/lr_model.pkl')
print("--lr_model.pkl salvo\n")

print("Treinando Naive Bayes...\n")
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
joblib.dump(nb, './models/nb_model.pkl')
print("--nb_model.pkl salvo\n")

print("Treinando Random Forest\n")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_vec, y_train)
joblib.dump(rf, './models/rf_model.pkl')
print("--rf_model.pkl salvo\n")

print("Treinando Support Vector Machine...\n")
svm = LinearSVC(random_state=42, dual=False)
svm.fit(X_train_vec, y_train)
joblib.dump(svm, './models/svm_model.pkl')
print("--svm_model.pkl salvo\n")

print("Treinando RNN (LSTM)...\n")

vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

rnn_checkpoint = ModelCheckpoint('./models/rnn_model.keras', monitor='val_loss', save_best_only=True)

model_rnn = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_LEN),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_rnn.fit(
    X_train_pad, y_train_nn,
    validation_data=(X_val_pad, y_val_nn),
    epochs=15, batch_size=64,
    callbacks=[early_stopping, rnn_checkpoint],
    verbose=0
)
print("--rnn_model.keras salvo\n")

print("Treinamento concluído. Todos os modelos foram salvos na pasta 'models'.\n")
