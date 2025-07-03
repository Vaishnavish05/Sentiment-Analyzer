import streamlit as st
import pickle
import requests
import io
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk

# Set NLTK download path to local folder (for Streamlit Cloud)
nltk.data.path.append("nltk_data")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------
# Page Setup
# ---------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# ---------------------
# Sidebar
# ---------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chat.png", width=80)
    st.markdown("### 👩‍💻 Built by Vaishnavi Sharma")
    st.markdown("[GitHub](https://github.com/Vaishnavish05) | [LinkedIn](https://www.linkedin.com/in/vaishnavish05/)")
    st.write("A Streamlit-powered sentiment analysis app using traditional ML models. Predicts sentiment as Positive, Negative, or Neutral.")

# ---------------------
# Text Preprocessing
# ---------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ---------------------
# Load Model from Google Drive
# ---------------------
@st.cache_resource
def load_model():
    file_id = "1Ck6GXEidnnw0jEmzXbCB4YEqKkTOf44E"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    st.write(f"📦 Fetching model from: {url}")  # DEBUG LINE
    
    response = requests.get(url)
    st.write(f"📶 Status Code: {response.status_code}")  # DEBUG LINE

    if response.status_code != 200:
        st.error("❌ Failed to load model from Google Drive.")
        return None

    try:
        model = pickle.load(io.BytesIO(response.content))
        st.success("✅ Model loaded successfully!")  # DEBUG LINE
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()
# ---------------------
# UI
# ---------------------
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Drop your thoughts below, and I’ll decode the <b>mood</b> 💭📊</p>", unsafe_allow_html=True)
st.markdown("---")

user_input = st.text_area("📝 What's on your mind today?", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Text can't be empty!")
    elif model is None:
        st.error("⚠️ Model couldn't be loaded.")
    else:
        processed_text = preprocess_text(user_input)
        prediction = model.predict([processed_text])[0]
        sentiment_map = {0: "😠 Negative", 2: "😐 Neutral", 4: "😊 Positive"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        st.success(f"Predicted Sentiment: **{sentiment}**")

        with st.expander("🔍 Show preprocessed text"):
            st.code(processed_text)
