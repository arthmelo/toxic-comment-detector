import streamlit as st
import joblib
import spacy
import re
import os
import pickle

import base64

RNN_THRESHOLD = 0.326

st.set_page_config(page_title="InstaCão", layout="centered", initial_sidebar_state="collapsed")

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
    elif name == "RNN (LSTM)":
        import tensorflow as tf
        return tf.keras.models.load_model('models/rnn_model.keras')
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
        
    elif model_name == "RNN (LSTM)":
        import tensorflow as tf
        tokenizer = get_tokenizer()
        if not tokenizer:
            return False
        sequence = tokenizer.texts_to_sequences([processed])
        padded = tf.keras.utils.pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        probability = model.predict(padded, verbose=0)[0][0]
        return probability >= RNN_THRESHOLD
        
    return False

if "comments" not in st.session_state:
    st.session_state.comments = []

st.markdown("""<style>
.stApp { background-color: #000000; color: #F5F5F5; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 5rem; padding-left: 0 !important; padding-right: 0 !important; max-width: 500px !important; }
.stMainBlockContainer { padding-left: 0 !important; padding-right: 0 !important; }
.stMarkdown { width: 100% !important; }
.stMarkdown > div { width: 100% !important; }
.insta-container { border: 1px solid #262626; border-radius: 4px; background-color: #000000; margin-bottom: 20px; padding: 0 !important; overflow: hidden; }
.insta-header { display: flex; align-items: center; padding: 12px 16px; }
.profile-pic { width: 32px; height: 32px; border-radius: 50%; object-fit: cover; margin-right: 10px; border: 1px solid #262626; }
.header-text { display: flex; flex-direction: column; justify-content: center; }
.username { font-weight: 600; font-size: 14px; margin: 0; line-height: 18px; color: #F5F5F5; }
.location { font-size: 12px; color: #A8A8A8; margin: 0; line-height: 16px; }
.options-icon { margin-left: auto; color: #F5F5F5; }
.post-image { width: 100%; height: auto; display: block; margin: 0; padding: 0; }
.action-icons { padding: 12px 16px; display: flex; justify-content: space-between; color: #F5F5F5; }
.action-icons-left svg { margin-right: 16px; cursor: pointer; }
.likes-count { padding: 0 16px; font-weight: 600; font-size: 14px; margin-bottom: 6px; color: #F5F5F5; }
.caption { padding: 0 16px; font-size: 14px; margin-bottom: 10px; color: #F5F5F5; }
.caption-username { font-weight: 600; margin-right: 6px; }
.comments-section { padding: 0 16px; font-size: 14px; margin-bottom: 12px; }
.comment-item { margin-bottom: 6px; line-height: 18px; }
.comment-username { font-weight: 600; margin-right: 6px; color: #F5F5F5; }
.censored-comment { color: #FF4A4A; font-style: italic; }
.normal-comment { color: #F5F5F5; }
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# Title and filter config
st.markdown("<h2 style='text-align: center; font-family: cursive; margin-bottom: 10px; margin-top: 0px;'>InstaCão</h2>", unsafe_allow_html=True)

model_options = [
    "Sem Filtro", 
    "Regressão Logística", 
    "Naive Bayes", 
    "Random Forest", 
    "SVM", 
    "RNN (LSTM)"
]

with st.expander("Filtro de Toxicidade"):
    selected_model = st.selectbox("Escolha o Filtro:", model_options, label_visibility="collapsed")
    if selected_model == "Sem Filtro":
        st.caption("O Filtro está DESLIGADO. Todos os comentários serão exibidos normalmente.")
    else:
        st.caption(f"Filtro ATIVO com **{selected_model}**. Comentários ofensivos serão censurados.")

# Image handling
image_path = "data/postagem.png"
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return ""

img_base64 = get_image_base64(image_path)
if img_base64:
    profile_img_src = f"data:image/png;base64,{img_base64}"
else:
    profile_img_src = "https://via.placeholder.com/150"

# Generate HTML Post
html_post = f"""<div class="insta-container">
<div class="insta-header">
<img src="{profile_img_src}" class="profile-pic">
<div class="header-text">
<p class="username">Snobi</p>
<p class="location">Ilha do Fundão, RJ</p>
</div>
<div class="options-icon">
<svg aria-label="Mais opções" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24"><circle cx="12" cy="12" r="1.5"></circle><circle cx="6" cy="12" r="1.5"></circle><circle cx="18" cy="12" r="1.5"></circle></svg>
</div>
</div>
<img src="{profile_img_src}" class="post-image">
<div class="action-icons">
<div class="action-icons-left">
<svg aria-label="Curtir" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24"><path d="M16.792 3.904A4.989 4.989 0 0 1 21.5 9.122c0 3.072-2.652 4.959-5.197 7.222-2.512 2.243-3.865 3.469-4.303 3.752-.477-.309-2.143-1.823-4.303-3.752C5.141 14.072 2.5 12.167 2.5 9.122a4.989 4.989 0 0 1 4.708-5.218 4.21 4.21 0 0 1 3.675 1.941c.84 1.175.98 1.763 1.12 1.763s.278-.588 1.11-1.766a4.17 4.17 0 0 1 3.679-1.938m0-2a6.14 6.14 0 0 0-4.896 2.397A6.12 6.12 0 0 0 7.208 1.904C3.8 1.904 1 4.542 1 9.122c0 4.135 3.32 6.438 6.44 9.155 2.19 1.907 3.51 3.02 4.14 3.627a1.002 1.002 0 0 0 1.41 0c.63-.607 1.95-1.72 4.14-3.627 3.12-2.717 6.44-5.02 6.44-9.155 0-4.58-2.8-7.218-6.208-7.218Z"></path></svg>
<svg aria-label="Comentar" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24"><path d="M20.656 17.008a9.993 9.993 0 1 0-3.59 3.615L22 22Z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></path></svg>
<svg aria-label="Compartilhar" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24"><line fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="2" x1="22" x2="9.218" y1="3" y2="10.083"></line><polygon fill="none" points="11.698 20.334 22 3.001 2 3.001 9.218 10.084 11.698 20.334" stroke="currentColor" stroke-linejoin="round" stroke-width="2"></polygon></svg>
</div>
<div>
<svg aria-label="Salvar" fill="currentColor" height="24" role="img" viewBox="0 0 24 24" width="24"><polygon fill="none" points="20 21 12 13.44 4 21 4 3 20 3 20 21" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></polygon></svg>
</div>
</div>
<div class="likes-count">2 curtidas</div>
<div class="caption"><span class="caption-username">Snobi</span>#novafotodeperfil ✨🐶</div>
<div class="comments-section">"""

for c in st.session_state.comments:
    if c["is_censored"]:
        html_post += f'<div class="comment-item"><span class="comment-username">usuario_anonimo</span><span class="censored-comment">{c["text"]}</span></div>'
    else:
        html_post += f'<div class="comment-item"><span class="comment-username">usuario_anonimo</span><span class="normal-comment">{c["text"]}</span></div>'

html_post += "</div></div>"

st.markdown(html_post, unsafe_allow_html=True)

# Chat Input at the bottom
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

    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
