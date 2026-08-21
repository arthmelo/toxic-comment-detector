import streamlit as st
import joblib
import spacy
import re
import os
import pickle

st.set_page_config(page_title="Instagrão", layout="centered")

@st.cache_resource
def get_spacy():
    try:
        return spacy.load("pt_core_news_sm")
    except OSError:
        return None

@st.cache_resource
def get_tfidf():
    return joblib.load('models/tfidf.pkl')

@st.cache_resource
def get_tokenizer():
    with open('models/tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

@st.cache_resource
def load_model_by_name(name):
    if name == "Regressão Logística":
        return joblib.load('models/lr_model.pkl')
    elif name == "Naive Bayes":
        return joblib.load('models/nb_model.pkl')
    elif name == "Random Forest":
        return joblib.load('models/rf_model.pkl')
    elif name == "SVM":
        return joblib.load('models/svm_model.pkl')
    elif name == "CNN":
        from tensorflow.keras.models import load_model
        return load_model('models/cnn_model.keras')
    return None

nlp = get_spacy()

def preProcessar(texto):
    if not nlp:
        return texto
    texto = str(texto).lower()
    texto = re.sub(r'[^a-záéíóúâêôãõüç\s]', ' ', texto)
    doc = nlp(texto)
    return " ".join([token.text for token in doc if not token.is_stop and token.text.strip()])

def is_toxic(comment, model_name):
    if model_name == "Sem Filtro":
        return False
        
    model = load_model_by_name(model_name)
    if not model:
        return False
        
    processed = preProcessar(comment)
    if not processed.strip():
        return False

    if model_name in ["Regressão Logística", "Naive Bayes", "Random Forest", "SVM"]:
        tfidf = get_tfidf()
        if not tfidf:
            return False
        vec = tfidf.transform([processed])
        
        prediction = model.predict(vec)[0]
        return prediction == 1
        
    elif model_name == "CNN":
        tokenizer = get_tokenizer()
        if not tokenizer:
            return False
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequence = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        probability = model.predict(padded, verbose=0)[0][0]
        return probability >= 0.5
        
    return False

if "comments" not in st.session_state:
    st.session_state.comments = []

st.title("Instagrão")

image_path = "data/postagem.png"

col1, col2 = st.columns([1, 8], vertical_alignment="center")
with col1:
    if os.path.exists(image_path):
        st.image(image_path, width=50)
    else:
        st.markdown("<h1 style='margin:0;'></h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h4 style='margin-bottom: 0px; margin-top: 0px; color: white;'>Snobi</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: gray; margin-top: -10px; font-size: 14px;'>Ilha do Fundão/ RJ</p>", unsafe_allow_html=True)

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
    st.markdown("<p style='color: white; font-size: 20px;'><b>Snobi</b> #novafotodeperfil</p>", unsafe_allow_html=True)
else:
    st.warning("Imagem da postagem não encontrada em 'data/postagem.png'.")

model_options = [
    "Sem Filtro", 
    "Regressão Logística", 
    "Naive Bayes", 
    "Random Forest", 
    "SVM", 
    "CNN"
]
selected_model = st.selectbox("Escolha o Filtro de Toxicidade:", model_options)

if selected_model == "Sem Filtro":
    st.warning("O Filtro está DESLIGADO. Todos os comentários serão exibidos normalmente.")
else:
    st.info(f"Filtro ATIVO com **{selected_model}**. Comentários ofensivos serão censurados.")

st.divider()
st.subheader("Comentários")

for c in st.session_state.comments:
    with st.chat_message("user"):
        if c["is_censored"]:
            st.error(c["text"])
        else:
            st.write(c["text"])

prompt = st.chat_input("Adicione um comentário...")

if prompt:
    censored = False
    display_text = prompt

    if selected_model != "Sem Filtro":
        if is_toxic(prompt, selected_model):
            censored = True
            display_text = "[Comentário censurado pelo filtro de toxicidade]"
            st.toast("Seu comentário violou as regras e foi censurado.")
    
    st.session_state.comments.append({
        "text": display_text,
        "is_censored": censored
    })

    with st.chat_message("user"):
        if censored:
            st.error(display_text)
        else:
            st.write(display_text)
