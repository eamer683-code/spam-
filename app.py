import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from arabic_data import get_all_arabic_data

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector | كاشف الرسائل المزعجة",
    page_icon="📧", layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title  { font-size:2.3rem; font-weight:700; color:#1a1a2e; text-align:center; margin-bottom:.2rem; }
.subtitle    { text-align:center; color:#666; font-size:1rem; margin-bottom:1.5rem; }
.spam-box    { background:linear-gradient(135deg,#ff4444,#cc0000); color:white;
               padding:1.4rem 2rem; border-radius:12px; text-align:center;
               font-size:1.4rem; font-weight:bold; margin:1rem 0; }
.ham-box     { background:linear-gradient(135deg,#00c853,#007e33); color:white;
               padding:1.4rem 2rem; border-radius:12px; text-align:center;
               font-size:1.4rem; font-weight:bold; margin:1rem 0; }
.warn-box    { background:#fff3cd; border-left:5px solid #ffc107;
               padding:1rem 1.5rem; border-radius:8px; color:#856404; margin-top:.5rem; }
.safe-box    { background:#d4edda; border-left:5px solid #28a745;
               padding:1rem 1.5rem; border-radius:8px; color:#155724; margin-top:.5rem; }
.kw-badge    { background:#ff4444; color:white; padding:.2rem .7rem;
               border-radius:20px; font-size:.85rem; font-weight:600; }
.history-row { background:#f8f9fa; border-radius:8px; padding:.6rem 1rem;
               margin:.3rem 0; border-left:4px solid #dee2e6; font-size:.9rem; }
.spam-hist   { border-left-color:#F44336; }
.ham-hist    { border-left-color:#4CAF50; }
</style>
""", unsafe_allow_html=True)

# ── Stopwords ─────────────────────────────────────────────────
EN_SW = set(['i','me','my','we','our','you','your','he','him','his','she','her','it','its',
    'they','them','their','what','which','who','this','that','these','those','am',
    'is','are','was','were','be','been','being','have','has','had','do','does',
    'did','a','an','the','and','but','if','or','as','of','at','by','for','with',
    'to','from','up','in','out','on','off','then','here','there','when','where',
    'how','all','both','more','most','no','not','only','so','than','too','very',
    'can','will','just','should','now'])
AR_SW = set(['في','من','إلى','على','عن','مع','هذا','هذه','ذلك','التي','الذي','وهو','وهي',
    'كان','كانت','يكون','تكون','هو','هي','هم','نحن','أنت','أنا','لكن','أو',
    'ثم','حتى','إذا','قد','لم','لن','ما','لا','إن','أن','بعد','قبل','كل',
    'بين','حول','خلال','عند','منذ','لدى','هل','كيف','أين','متى','لماذا','الى','عبر','غير','فقط'])

# ── Keywords ──────────────────────────────────────────────────
EN_SPAM_KW = [
    'winner','won','prize','free','claim','urgent','congratulations','click here',
    'limited offer','cash','lottery','selected','reward','guaranteed','exclusive',
    'discount','act now','call now','order now','sign up','verify','suspended',
    'account suspended','credit','loan','investment','earn money','make money',
    'work from home','no experience','risk free','casino','jackpot','bitcoin',
    'double your','bank details','personal details','buy now','get rich',
    'apply now','special offer','you have been selected','you have won',
]
AR_SPAM_KW = [
    'مبروك','فزت','جائزه','جوائز','مجانا','مجاني','خصم','عرض خاص','اتصل الان',
    'سارع','محدود','حصري','ربح','اكسب','استثمر','قرض','تمويل','بدون فوائد',
    'موافقه','تهانينا','تم اختيارك','عاجل','فوري','تحقق','بياناتك','حسابك',
    'اختراق','ايقاف','تجديد','اشتراكك','انتهي','انقر هنا','اضغط هنا',
    'سجل الان','اشترك الان','هديه مجانيه','رحله مجانيه','ايفون مجاني',
    'لابتوب مجاني','ضاعف اموالك','ارسل بياناتك','رقم حسابك','تسوق الان',
    'اطلب الان','توصيل مجاني','ربح سريع','فرصه ذهبيه','استثمار مضمون',
    'عائد مضمون','كسب من المنزل','بدون خبره','تجميد حسابك','نشاط مشبوه',
    'تحديث عاجل','انتهت صلاحيته','واتساب','تلجرام','مليونير','فوركس',
]

def keyword_check(text):
    t = text.lower()
    t_n = re.sub(r'[أإآا]','ا', re.sub(r'ة','ه', re.sub(r'ى','ي',
          re.sub(r'[\u064B-\u065F\u0670]','', t))))
    for kw in EN_SPAM_KW:
        if kw in t: return True, kw
    for kw in AR_SPAM_KW:
        kw_n = re.sub(r'[أإآا]','ا', re.sub(r'ة','ه', re.sub(r'ى','ي', kw)))
        if kw_n in t_n: return True, kw
    return False, None

def preprocess_text(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\u0600-\u06FFa-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split()
              if w not in EN_SW and w not in AR_SW and len(w) > 2]
    return ' '.join(tokens)

# ── Train model ───────────────────────────────────────────────
@st.cache_resource
def train_model():
    try:
        df_en = pd.read_csv('spam.csv', encoding='latin-1')
        df_en = df_en[['v1','v2']]
        df_en.columns = ['label','message']
    except Exception:
        df_en = pd.DataFrame(columns=['label','message'])

    ar_data = get_all_arabic_data()
    df_ar   = pd.DataFrame(ar_data, columns=['message','label'])
    df = pd.concat([df_en, df_ar], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['clean']     = df['message'].apply(preprocess_text)
    df['label_num'] = df['label'].map({'ham':0,'spam':1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_tr  = tfidf.fit_transform(X_train)
    X_te  = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    metrics = {
        'accuracy' : round(accuracy_score(y_test, y_pred)*100, 2),
        'precision': round(precision_score(y_test, y_pred)*100, 2),
        'recall'   : round(recall_score(y_test, y_pred)*100, 2),
        'f1'       : round(f1_score(y_test, y_pred)*100, 2),
        'cm'       : confusion_matrix(y_test, y_pred),
        'total'    : len(df),
        'en_count' : len(df_en),
        'ar_count' : len(df_ar),
        'spam_count': int(df['label_num'].sum()),
        'ham_count' : int((df['label_num']==0).sum()),
    }
    return model, tfidf, metrics

def predict(text, model, tfidf):
    kw_spam, matched_kw = keyword_check(text)
    cleaned = preprocess_text(text)
    vec     = tfidf.transform([cleaned])
    ml_pred = model.predict(vec)[0]
    ml_prob = model.predict_proba(vec)[0]
    ml_conf = ml_prob[ml_pred] * 100
    final   = 1 if (kw_spam or ml_pred == 1) else 0
    return final, ml_pred, ml_conf, matched_kw

# ── Init session state ────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'msg_input' not in st.session_state:
    st.session_state.msg_input = ''

# ── Load model ────────────────────────────────────────────────
st.markdown('<p class="main-title">📧 Email Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multilingual: English 🇬🇧 + Arabic 🇸🇦 &nbsp;|&nbsp; Hybrid Detection (Keyword + Naive Bayes)</p>',
            unsafe_allow_html=True)

with st.spinner('🔄 Training model on 5,888+ messages...'):
    model, tfidf, metrics = train_model()
st.success(f'✅ Model ready! Trained on {metrics["total"]:,} messages (EN: {metrics["en_count"]:,} + AR: {metrics["ar_count"]:,})')
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🔍 Detect Spam', '📜 History', '📊 Model Performance'])

# ══ TAB 1: DETECT ════════════════════════════════════════════
with tab1:
    st.subheader('اكتب رسالتك بالعربي أو الإنجليزي | Type your message:')

    # Quick examples
    st.markdown('**أمثلة سريعة | Quick examples:**')
    c1, c2, c3, c4 = st.columns(4)
    examples = {
        '🚨 Spam EN': "WINNER!! You've been selected to receive $1000. Call now to claim your prize!",
        '🚨 Spam AR (عروض)': "عرض خاص! احصل على خصم تسعين بالمئة على جميع المنتجات. اشترِ الآن!",
        '🚨 Spam AR (بنكي)': "عاجل: تم اختراق حسابك البنكي. تحقق من بياناتك الآن فوراً!",
        '🚨 Spam AR (واتساب)': "واتساب: حسابك سيُحذف خلال 24 ساعة. انقر الرابط لتفعيله الآن.",
        '✅ Ham EN': "Hey, are you coming to the team meeting at 3pm today? Please confirm.",
        '✅ Ham AR': "هل ستحضر الاجتماع غداً؟ أرسل لي التقرير من فضلك قبل نهاية اليوم.",
        '🚨 Spam AR (نصب)': "مبروك! لقد فزت بجائزة كبيرة. أرسل بياناتك الآن للمطالبة بجائزتك!",
        '✅ Ham AR': "شكراً على مساعدتك بالأمس كثيراً. سأكون في المكتب من الساعة التاسعة.",
    }
    ex_list = list(examples.items())
    for i, col in enumerate([c1, c2, c3, c4]):
        if col.button(ex_list[i][0]):
            st.session_state.msg_input = ex_list[i][1]
    c5, c6, c7, c8 = st.columns(4)
    for i, col in enumerate([c5, c6, c7, c8]):
        if i+4 < len(ex_list):
            if col.button(ex_list[i+4][0], key=f'ex_{i+4}'):
                st.session_state.msg_input = ex_list[i+4][1]

    user_input = st.text_area(
        'الرسالة | Message:',
        value=st.session_state.msg_input,
        height=140,
        placeholder='اكتب رسالتك هنا... / Type your message here...'
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    analyze = col_btn1.button('🔍 تحليل الرسالة | Analyze', type='primary', use_container_width=True)
    clear   = col_btn2.button('🗑️ Clear', use_container_width=True)
    if clear:
        st.session_state.msg_input = ''
        st.rerun()

    if analyze:
        if user_input.strip():
            with st.spinner('جاري التحليل... | Analyzing...'):
                time.sleep(0.3)
                final, ml_pred, ml_conf, matched_kw = predict(user_input, model, tfidf)

            spam_conf = ml_conf if ml_pred == 1 else 100 - ml_conf
            ham_conf  = 100 - spam_conf

            if final == 1:
                st.markdown('<div class="spam-box">🚨 SPAM DETECTED | رسالة مزعجة</div>',
                            unsafe_allow_html=True)
                kw_html = f'<br>🔑 Matched keyword: <span class="kw-badge">{matched_kw}</span>' if matched_kw else ''
                st.markdown(f"""
                <div class="warn-box">
                    <strong>⚠️ WARNING | تحذير</strong>{kw_html}<br><br>
                    🔴 This message appears to be <strong>SPAM</strong> ({spam_conf:.1f}% confidence)<br>
                    🔴 هذه الرسالة تبدو <strong>مزعجة أو احتيالية</strong><br><br>
                    ❌ Do NOT click any links &nbsp;|&nbsp; لا تضغط على أي روابط<br>
                    ❌ Do NOT share personal info &nbsp;|&nbsp; لا تشارك بياناتك الشخصية<br>
                    ❌ Do NOT reply &nbsp;|&nbsp; لا ترد على هذه الرسالة<br>
                    🗑️ Mark as spam and delete &nbsp;|&nbsp; ضعها في البريد المزعج واحذفها
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="ham-box">✅ SAFE — Not Spam | رسالة آمنة</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="safe-box">
                    <strong>✅ This message looks safe! | الرسالة تبدو آمنة!</strong><br>
                    Classified as <strong>Ham</strong> with {ham_conf:.1f}% confidence.<br>
                    تم تصنيفها كرسالة طبيعية بنسبة ثقة {ham_conf:.1f}%
                </div>""", unsafe_allow_html=True)

            # Confidence bars
            st.markdown('#### نسبة الثقة | Confidence')
            ca, cb = st.columns(2)
            with ca:
                st.metric('🚨 Spam Probability', f'{spam_conf:.1f}%')
                st.progress(spam_conf / 100)
            with cb:
                st.metric('✅ Ham Probability', f'{ham_conf:.1f}%')
                st.progress(ham_conf / 100)

            # Save to history
            st.session_state.history.insert(0, {
                'time'    : datetime.now().strftime('%H:%M:%S'),
                'message' : user_input[:80] + ('...' if len(user_input) > 80 else ''),
                'result'  : 'SPAM' if final == 1 else 'HAM',
                'conf'    : f'{spam_conf:.1f}%' if final == 1 else f'{ham_conf:.1f}%',
                'keyword' : matched_kw or '-',
            })
        else:
            st.warning('⚠️ من فضلك اكتب رسالة أولاً | Please enter a message first.')

# ══ TAB 2: HISTORY ════════════════════════════════════════════
with tab2:
    st.subheader('📜 Spam Detection History | سجل التحليلات')
    if not st.session_state.history:
        st.info('No messages analyzed yet. | لم يتم تحليل أي رسائل بعد.')
    else:
        col_s, col_h = st.columns(2)
        spam_count = sum(1 for h in st.session_state.history if h['result'] == 'SPAM')
        ham_count  = len(st.session_state.history) - spam_count
        col_s.metric('🚨 Spam Detected', spam_count)
        col_h.metric('✅ Ham Messages',  ham_count)
        st.divider()
        for h in st.session_state.history:
            css = 'spam-hist' if h['result'] == 'SPAM' else 'ham-hist'
            icon = '🚨' if h['result'] == 'SPAM' else '✅'
            kw_text = f" | 🔑 {h['keyword']}" if h['keyword'] != '-' else ''
            st.markdown(
                f'<div class="history-row {css}">'
                f'{icon} <strong>{h["result"]}</strong> ({h["conf"]}) '
                f'— {h["time"]}{kw_text}<br>'
                f'<small>{h["message"]}</small></div>',
                unsafe_allow_html=True
            )
        if st.button('🗑️ Clear History | مسح السجل'):
            st.session_state.history = []
            st.rerun()

# ══ TAB 3: PERFORMANCE ════════════════════════════════════════
with tab3:
    st.subheader('📊 Model Evaluation Results')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('🎯 Accuracy',  f"{metrics['accuracy']}%")
    c2.metric('📌 Precision', f"{metrics['precision']}%")
    c3.metric('🔍 Recall',    f"{metrics['recall']}%")
    c4.metric('⚖️ F1-Score',  f"{metrics['f1']}%")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('#### Confusion Matrix')
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'], ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout(); st.pyplot(fig)

    with col_r:
        st.markdown('#### Dataset & Model Info')
        st.markdown(f"""
| Detail | Value |
|--------|-------|
| **Total Dataset** | {metrics['total']:,} messages |
| **English (SMS Spam)** | {metrics['en_count']:,} messages |
| **Arabic** | {metrics['ar_count']:,} messages |
| **Total Spam** | {metrics['spam_count']:,} |
| **Total Ham** | {metrics['ham_count']:,} |
| **Algorithm** | Multinomial Naive Bayes |
| **Vectorizer** | TF-IDF (5000 features) |
| **N-grams** | Unigrams + Bigrams |
| **Train/Test Split** | 80% / 20% |
| **Detection Method** | Hybrid (Keyword + ML) |
| **Languages** | English 🇬🇧 + Arabic 🇸🇦 |
        """)

    st.divider()
    st.markdown('#### Arabic Spam Categories | فئات الـ Spam العربي')
    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)
    cat_col1.info('🛍️ **عروض وهمية**\nFake Offers\n(50 messages)')
    cat_col2.warning('🎭 **رسائل نصب**\nScam Messages\n(51 messages)')
    cat_col3.error('🏦 **بنكية مزيفة**\nFake Banking\n(51 messages)')
    cat_col4.info('📱 **واتساب / تلجرام**\nFake Social\n(50 messages)')

    st.divider()
    st.info("""
    **How Hybrid Detection Works:**
    
    1. **Keyword Check** — Scans for known spam keywords in EN+AR. Instant flag.
    2. **ML Model (Naive Bayes + TF-IDF)** — Classifies based on learned patterns.
    3. **Final Decision** — SPAM if *either* method flags it. Maximizes recall.
    """)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#999;font-size:.82rem;">'
    'Intelligent Programming Project · Email Spam Detection · EN 🇬🇧 + AR 🇸🇦 · Hybrid Detection'
    '</p>', unsafe_allow_html=True
)
