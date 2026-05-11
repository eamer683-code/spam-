import streamlit as st
import pandas as pd
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

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector | كاشف الرسائل المزعجة",
    page_icon="📧",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 700;
        color: #1a1a2e; text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #666;
        font-size: 1rem; margin-bottom: 2rem;
    }
    .spam-box {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem;
        font-weight: bold; margin: 1rem 0;
    }
    .ham-box {
        background: linear-gradient(135deg, #00c853, #007e33);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        text-align: center; font-size: 1.4rem;
        font-weight: bold; margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem; border-radius: 8px;
        color: #856404; margin-top: 0.5rem;
        direction: ltr;
    }
    .safe-box {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 1rem 1.5rem; border-radius: 8px;
        color: #155724; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Stopwords ─────────────────────────────────────────────────
EN_STOPWORDS = set([
    'i','me','my','we','our','you','your','he','him','his','she','her','it','its',
    'they','them','their','what','which','who','this','that','these','those','am',
    'is','are','was','were','be','been','being','have','has','had','do','does',
    'did','a','an','the','and','but','if','or','as','of','at','by','for','with',
    'to','from','up','in','out','on','off','then','here','there','when','where',
    'how','all','both','more','most','no','not','only','so','than','too','very',
    'can','will','just','should','now'
])

AR_STOPWORDS = set([
    'في','من','إلى','على','عن','مع','هذا','هذه','ذلك','التي','الذي','وهو','وهي',
    'كان','كانت','يكون','تكون','هو','هي','هم','نحن','أنت','أنا','لكن','أو',
    'ثم','حتى','إذا','قد','لم','لن','ما','لا','إن','أن','بعد','قبل','كل',
    'بين','حول','خلال','عند','منذ','لدى','هل','كيف','أين','متى','لماذا'
])

# ── Preprocessing ─────────────────────────────────────────────
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
    tokens = text.split()
    tokens = [w for w in tokens if w not in EN_STOPWORDS
              and w not in AR_STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Arabic data ───────────────────────────────────────────────
ARABIC_DATA = [
    ('مبروك! لقد فزت بجائزة قيمتها عشرة آلاف جنيه. اتصل الآن للمطالبة بجائزتك!', 'spam'),
    ('عرض خاص! احصل على خصم 90% على جميع المنتجات. اشترِ الآن قبل انتهاء العرض!', 'spam'),
    ('تهانينا! رقم هاتفك فاز بسيارة جديدة. أرسل بياناتك الشخصية للمطالبة بالجائزة.', 'spam'),
    ('عاجل: تم اختراق حسابك البنكي. تحقق من بياناتك الآن على الرابط التالي.', 'spam'),
    ('اربح المال من المنزل! لا خبرة مطلوبة. انضم الآن واكسب خمسة آلاف جنيه أسبوعياً.', 'spam'),
    ('أدوية رخيصة عبر الإنترنت! أفضل الأسعار. اطلب الآن وتوصيل مجاني.', 'spam'),
    ('فرصة استثمارية ذهبية! ضاعف أموالك في أسبوع. تواصل معنا الآن.', 'spam'),
    ('تنبيه: سيتم إيقاف حسابك. تحقق من بياناتك فوراً لتجنب الإيقاف.', 'spam'),
    ('مجاناً! اشترك الآن واحصل على هاتف آيفون مجاناً. العرض محدود!', 'spam'),
    ('لقد تم اختيارك للحصول على قرض بدون فوائد. اتصل بنا الآن للاستفادة.', 'spam'),
    ('عرض لا يُفوَّت! اشترِ الآن بالتقسيط بدون فوائد. سارع قبل نفاد الكمية!', 'spam'),
    ('مبروك عليك الفوز! أرسل اسمك ورقم حسابك للحصول على الجائزة فوراً.', 'spam'),
    ('احصل على عضوية VIP مجانية لمدة شهر! سجل الآن ببياناتك الشخصية.', 'spam'),
    ('تحذير: اشتراكك انتهى. جدد الآن وإلا سيتم حذف حسابك نهائياً!', 'spam'),
    ('ربح سريع! استثمر مائة جنيه واحصل على ألف جنيه في يوم واحد فقط.', 'spam'),
    ('رسالة مهمة: فزت في السحب على الجوائز. أرسل بياناتك لاستلام الجائزة.', 'spam'),
    ('عزيزي العميل، تم الموافقة على قرضك. تواصل معنا لاستلام المبلغ فوراً.', 'spam'),
    ('خصم سبعين بالمئة على كل شيء! تسوق الآن قبل انتهاء التخفيضات الكبرى.', 'spam'),
    ('رابط تسجيل الدخول انتهت صلاحيته. انقر هنا لتحديث بياناتك.', 'spam'),
    ('عرض خاص لك فقط! احصل على رحلة مجانية إلى دبي. سجل الآن!', 'spam'),
    ('تم اختيارك عشوائياً للفوز بجهاز لابتوب. تواصل معنا لاستلامه.', 'spam'),
    ('إشعار بنكي: حسابك يحتاج تحديثاً عاجلاً. ادخل بياناتك هنا الآن.', 'spam'),
    ('جوائز يومية! أجب على سؤال بسيط واربح جائزة نقدية كبيرة الآن.', 'spam'),
    ('اشترك في خدمتنا المميزة بخمسة جنيه فقط يومياً واحصل على عروض حصرية!', 'spam'),
    ('تفضل بزيارة موقعنا واحصل على هدية مجانية عند أول طلب لك اليوم.', 'spam'),
    ('هل ستحضر الاجتماع غداً الساعة التاسعة صباحاً؟', 'ham'),
    ('أرسل لي التقرير عندما تنتهي منه من فضلك.', 'ham'),
    ('كل عام وأنت بخير! أتمنى لك يوم ميلاد سعيد.', 'ham'),
    ('هل يمكنك مساعدتي في مراجعة المشروع قبل التسليم؟', 'ham'),
    ('سأتأخر قليلاً في الوصول، ابدأوا الاجتماع بدوني.', 'ham'),
    ('شكراً جزيلاً على مساعدتك في المهمة الأخيرة.', 'ham'),
    ('هل يمكنني تغيير موعد اللقاء إلى يوم الخميس؟', 'ham'),
    ('حجزت مطعم الساعة السابعة مساءً. أراك هناك!', 'ham'),
    ('الطقس جميل اليوم، هل تريد الذهاب للمشي في الحديقة؟', 'ham'),
    ('موعد الطبيب غداً الساعة الثانية ظهراً. لا تنسَ!', 'ham'),
    ('انتهيت من قراءة الكتاب الذي أوصيت به. كان رائعاً!', 'ham'),
    ('هل أنت متاح نهاية هذا الأسبوع؟ نريد الخروج معاً.', 'ham'),
    ('أحتاج عنوانك لإرسال الطرد إليك.', 'ham'),
    ('صباح الخير! أتمنى أن تكون بخير اليوم.', 'ham'),
    ('تم نقل الاجتماع إلى الساعة الثالثة عصراً. عدّل جدولك.', 'ham'),
    ('سأتصل بك عندما أصل إلى المطار.', 'ham'),
    ('عرضي التقديمي سار بشكل ممتاز اليوم. أنا سعيد جداً!', 'ham'),
    ('أخبرني إذا كنت تحتاج مساعدة في الانتقال إلى المنزل الجديد.', 'ham'),
    ('كيف كانت إجازتك؟ أريد أن أسمع عنها.', 'ham'),
    ('نسيت مظلتي عندك. هل يمكنني أخذها لاحقاً؟', 'ham'),
    ('هل يمكنك مراجعة سيرتي الذاتية قبل إرسالها؟', 'ham'),
    ('لدينا غداء جماعي يوم الجمعة. هل ستأتي معنا؟', 'ham'),
    ('أرسلت لك الملفات على البريد الإلكتروني. تفضل بالمراجعة.', 'ham'),
    ('الأطفال يسألون إذا كان يمكنهم المبيت عندكم الليلة.', 'ham'),
    ('تذكر أن لديك اجتماع مهم الأسبوع القادم مع الفريق.', 'ham'),
]

# ── Train model ───────────────────────────────────────────────
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    except Exception:
        df = pd.DataFrame(columns=['label', 'message'])

    arabic_df = pd.DataFrame(ARABIC_DATA, columns=['message', 'label'])
    df = pd.concat([df, arabic_df], ignore_index=True)

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
st.markdown(
    '<p class="subtitle">Multilingual: English 🇬🇧 + Arabic 🇸🇦 &nbsp;|&nbsp; Naive Bayes + TF-IDF</p>',
    unsafe_allow_html=True
)

with st.spinner('🔄 Training model...'):
    model, tfidf, metrics = train_model()

st.success('✅ Model ready!')
st.divider()

tab1, tab2 = st.tabs(['🔍 Detect Spam | كشف الرسائل', '📊 Model Performance'])

# ── Tab 1 ─────────────────────────────────────────────────────
with tab1:
    st.subheader('Type your message in English or Arabic:')
    st.caption('اكتب رسالتك بالإنجليزي أو العربي')

    # Quick examples
    st.markdown('**Quick examples | أمثلة سريعة:**')
    c1, c2, c3, c4 = st.columns(4)
    examples = {
        '🚨 Spam EN': "WINNER!! You won $1000. Call now to claim your prize!",
        '🚨 Spam AR': "مبروك! فزت بجائزة كبيرة. أرسل بياناتك الآن للمطالبة بجائزتك!",
        '✅ Ham EN':  "Hey, are you coming to the meeting at 3pm today?",
        '✅ Ham AR':  "هل ستحضر الاجتماع غداً؟ أرسل لي التقرير من فضلك.",
    }
    if c1.button('🚨 Spam EN'): st.session_state['msg'] = examples['🚨 Spam EN']
    if c2.button('🚨 Spam AR'): st.session_state['msg'] = examples['🚨 Spam AR']
    if c3.button('✅ Ham EN'):  st.session_state['msg'] = examples['✅ Ham EN']
    if c4.button('✅ Ham AR'):  st.session_state['msg'] = examples['✅ Ham AR']

    default = st.session_state.get('msg', '')
    user_input = st.text_area(
        'Message | الرسالة:',
        value=default,
        height=150,
        placeholder='Type here... / اكتب هنا...'
    )

    if st.button('🔍 Analyze | تحليل', type='primary', use_container_width=True):
        if user_input.strip():
            with st.spinner('Analyzing... | جاري التحليل...'):
                time.sleep(0.4)
                pred, confidence = predict(user_input, model, tfidf)

            if pred == 1:
                st.markdown('<div class="spam-box">🚨 SPAM DETECTED | رسالة مزعجة</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ WARNING | تحذير</strong><br><br>
                    🔴 This email appears to be <strong>SPAM</strong> ({confidence:.1f}% confidence)<br>
                    🔴 هذه الرسالة تبدو <strong>مزعجة أو احتيالية</strong> بنسبة {confidence:.1f}%<br><br>
                    ❌ Do NOT click any links | لا تضغط على أي روابط<br>
                    ❌ Do NOT share personal info | لا تشارك بياناتك الشخصية<br>
                    ❌ Do NOT reply | لا ترد على هذه الرسالة<br>
                    🗑️ Mark as spam and delete | ضعها في البريد المزعج واحذفها
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="ham-box">✅ SAFE — Not Spam | رسالة آمنة</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="safe-box">
                    <strong>✅ This message looks safe! | الرسالة تبدو آمنة!</strong><br>
                    Classified as <strong>Ham</strong> with {confidence:.1f}% confidence.<br>
                    تم تصنيفها كرسالة عادية بنسبة ثقة {confidence:.1f}%
                </div>
                """, unsafe_allow_html=True)

            # Confidence bars
            st.markdown('#### Confidence | نسبة الثقة')
            col_a, col_b = st.columns(2)
            spam_conf = confidence if pred == 1 else 100 - confidence
            ham_conf  = 100 - spam_conf
            with col_a:
                st.metric('🚨 Spam', f'{spam_conf:.1f}%')
                st.progress(spam_conf / 100)
            with col_b:
                st.metric('✅ Ham', f'{ham_conf:.1f}%')
                st.progress(ham_conf / 100)
        else:
            st.warning('⚠️ Please enter a message first. | من فضلك اكتب رسالة أولاً.')

# ── Tab 2 ─────────────────────────────────────────────────────
with tab2:
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
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.markdown('#### Model Info')
        st.markdown(f"""
        | Detail | Value |
        |--------|-------|
        | **Dataset** | {metrics['dataset_size']:,} messages |
        | **Languages** | English 🇬🇧 + Arabic 🇸🇦 |
        | **Algorithm** | Multinomial Naive Bayes |
        | **Vectorizer** | TF-IDF (5000 features) |
        | **N-grams** | Unigrams + Bigrams |
        | **Train/Test** | 80% / 20% |
        | **Bonus** | ✅ Multilingual + Warning Messages + UI |
        """)

st.divider()
st.markdown(
    '<p style="text-align:center;color:#999;font-size:0.85rem;">'
    'Intelligent Programming Project · Email Spam Detection · EN 🇬🇧 + AR 🇸🇦'
    '</p>',
    unsafe_allow_html=True
)
