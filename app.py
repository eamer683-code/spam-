import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .spam-box {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .ham-box {
        background: linear-gradient(135deg, #00c853, #007e33);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #856404;
        margin-top: 0.5rem;
    }
    .safe-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #155724;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Stopwords ─────────────────────────────────────────────────
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it','its',
    'itself','they','them','their','theirs','themselves','what','which','who','whom',
    'this','that','these','those','am','is','are','was','were','be','been','being',
    'have','has','had','having','do','does','did','doing','a','an','the','and','but',
    'if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below',
    'to','from','up','down','in','out','on','off','over','under','again','further',
    'then','once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own','same',
    'so','than','too','very','s','t','can','will','just','don','should','now'
])

# ── Text preprocessing ────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Train model (cached) ──────────────────────────────────────
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    except Exception:
        # Fallback sample dataset
        spam_msgs = [
            "WINNER!! You have been selected to receive a $1000 prize. Call now!",
            "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "Congratulations! You won a free iPhone. Click here to claim your prize.",
            "URGENT: Your account has been compromised. Verify now at fake-bank.com",
            "Get rich quick! Make $5000 a week from home. No experience needed.",
            "You have won a lottery! Send your bank details to claim your prize.",
            "Buy cheap meds online! Viagra, Cialis at lowest prices. Order now!",
            "Hot singles in your area want to meet you. Click here!",
            "Your mobile number has won $500,000. Reply YES to claim.",
            "Limited time offer! 90% discount on all products. Shop now!",
            "Dear customer, your loan is approved. Click to receive $10,000.",
            "Act now! Exclusive deal expires tonight. Call 0800-123456.",
            "You are selected for a free holiday. Text HOLIDAY to 80488.",
            "Claim your free gift now! Reply with your personal details.",
            "Earn money from home. No investment required. Join today!",
            "Alert: Your account will be suspended. Verify at spam-link.com",
            "Win a brand new car! Just answer this simple question.",
            "Special offer just for you! Buy now and save 70%.",
            "FREE ringtones! Text RING to 12345. Only $1.50/week.",
            "Meet hot girls in your area tonight! Text MEET to 62468.",
        ] * 5
        ham_msgs = [
            "Hey, are you coming to the party tonight?",
            "I will be late for the meeting. Please start without me.",
            "Can you pick up some milk on your way home?",
            "The project deadline is next Friday. Are we on track?",
            "Happy birthday! Hope you have a wonderful day.",
            "Did you watch the game last night? It was amazing!",
            "I am stuck in traffic. Will be there in 30 minutes.",
            "Can we reschedule our lunch to Thursday?",
            "Thanks for your help with the assignment yesterday.",
            "The kids have a school play on Wednesday evening.",
            "Are you free this weekend? We should catch up.",
            "I need your address to send you the package.",
            "Good morning! Hope you slept well.",
            "The meeting has been moved to 3pm. Please update your calendar.",
            "Can you send me the report when you get a chance?",
            "I will call you when I land at the airport.",
            "What do you want for dinner tonight?",
            "The presentation went really well today!",
            "Let me know if you need any help with the move.",
            "I booked the restaurant for 7pm. See you there!",
        ] * 5

        import random
        random.seed(42)
        data = [{'message': m, 'label': 'spam'} for m in spam_msgs] + \
               [{'message': m, 'label': 'ham'} for m in ham_msgs]
        random.shuffle(data)
        df = pd.DataFrame(data)

    df['clean_text'] = df['message'].apply(preprocess_text)
    df['label_num']  = df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'f1':        round(f1_score(y_test, y_pred) * 100, 2),
        'cm':        confusion_matrix(y_test, y_pred),
        'dataset_size': len(df),
    }

    return model, tfidf, metrics


def predict(text, model, tfidf):
    cleaned = preprocess_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba[pred] * 100


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">📧 Email Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Programming Project — Naive Bayes + TF-IDF</p>',
            unsafe_allow_html=True)

# Load model
with st.spinner('🔄 Training model...'):
    model, tfidf, metrics = train_model()

st.success('✅ Model ready!')
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(['🔍 Detect Spam', '📊 Model Performance'])

# ── Tab 1: Detect ─────────────────────────────────────────────
with tab1:
    st.subheader('Type or paste an email to check:')

    # Quick test buttons
    st.markdown('**Quick test examples:**')
    col1, col2, col3, col4 = st.columns(4)

    examples = {
        '🚨 Spam #1': "WINNER!! You have been selected to receive a $1000 prize. Call now to claim your reward!",
        '🚨 Spam #2': "Congratulations! You won a free iPhone. Click here to claim your prize. Limited time!",
        '✅ Ham #1':  "Hey, are you coming to the team meeting at 3pm today? Please confirm.",
        '✅ Ham #2':  "Can you send me the project report by tomorrow morning? Thanks.",
    }

    if col1.button('🚨 Spam #1'): st.session_state['email_input'] = examples['🚨 Spam #1']
    if col2.button('🚨 Spam #2'): st.session_state['email_input'] = examples['🚨 Spam #2']
    if col3.button('✅ Ham #1'):  st.session_state['email_input'] = examples['✅ Ham #1']
    if col4.button('✅ Ham #2'):  st.session_state['email_input'] = examples['✅ Ham #2']

    default_text = st.session_state.get('email_input', '')
    email_input = st.text_area(
        'Email Content:',
        value=default_text,
        height=160,
        placeholder='Type your email message here...'
    )

    if st.button('🔍 Analyze Email', type='primary', use_container_width=True):
        if email_input.strip():
            with st.spinner('Analyzing...'):
                time.sleep(0.4)
                pred, confidence = predict(email_input, model, tfidf)

            if pred == 1:
                st.markdown('<div class="spam-box">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ WARNING!</strong><br>
                    This email appears to be <strong>SPAM</strong> with <strong>{confidence:.1f}% confidence</strong>.<br><br>
                    🔴 <strong>Do NOT</strong> click any links in this email.<br>
                    🔴 <strong>Do NOT</strong> share personal or financial information.<br>
                    🔴 <strong>Do NOT</strong> reply to this message.<br>
                    🔴 Mark it as spam and delete it immediately.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="ham-box">✅ SAFE — Not Spam (Ham)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="safe-box">
                    <strong>✅ This email looks safe!</strong><br>
                    Classified as <strong>Ham</strong> with <strong>{confidence:.1f}% confidence</strong>.<br>
                    This message does not appear to contain spam content.
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown('#### Confidence')
            col_a, col_b = st.columns(2)
            with col_a:
                spam_conf = confidence if pred == 1 else 100 - confidence
                st.metric('🚨 Spam Probability', f'{spam_conf:.1f}%')
                st.progress(spam_conf / 100)
            with col_b:
                ham_conf = 100 - spam_conf
                st.metric('✅ Ham Probability', f'{ham_conf:.1f}%')
                st.progress(ham_conf / 100)
        else:
            st.warning('⚠️ Please enter an email message first.')

# ── Tab 2: Performance ────────────────────────────────────────
with tab2:
    st.subheader('📊 Model Evaluation Results')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('🎯 Accuracy',  f"{metrics['accuracy']}%")
    col2.metric('📌 Precision', f"{metrics['precision']}%")
    col3.metric('🔍 Recall',    f"{metrics['recall']}%")
    col4.metric('⚖️ F1-Score',  f"{metrics['f1']}%")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('#### Confusion Matrix')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.markdown('#### Model Information')
        st.markdown(f"""
        | Detail | Value |
        |--------|-------|
        | **Dataset Size** | {metrics['dataset_size']:,} messages |
        | **Algorithm** | Multinomial Naive Bayes |
        | **Vectorizer** | TF-IDF (5000 features) |
        | **N-grams** | Unigrams + Bigrams |
        | **Train/Test Split** | 80% / 20% |
        | **Preprocessing** | Lowercase, remove URLs, stopwords |
        """)

    st.divider()
    st.markdown('#### About This Project')
    st.info("""
    **Email Spam Detection** uses Natural Language Processing (NLP) to classify
    messages as spam or legitimate (ham).

    **Pipeline:**
    1. 📥 Load dataset
    2. 🧹 Text preprocessing (clean, normalize)
    3. 🔢 TF-IDF vectorization (text → numbers)
    4. 🤖 Train Naive Bayes classifier
    5. 📊 Evaluate with accuracy, precision, recall, F1
    6. 🔍 Predict new emails with warning messages
    """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center; color:#999; font-size:0.85rem;">'
    'Intelligent Programming Project · Email Spam Detection · Naive Bayes + TF-IDF'
    '</p>',
    unsafe_allow_html=True
)
