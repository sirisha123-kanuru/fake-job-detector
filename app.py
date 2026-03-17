import streamlit as st
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
from predict import predict_job

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake Job Posting Detector")
st.caption("Paste any job posting to check if it's real or a scam")
st.markdown("---")

# Input fields
col1, col2 = st.columns([2,1])
with col1:
    title = st.text_input(
        "Job title",
        placeholder="e.g. Software Engineer at Google"
    )
with col2:
    company = st.text_input(
        "Company name",
        placeholder="e.g. Google"
    )

description = st.text_area(
    "Job description",
    height=180,
    placeholder="Paste the full job description here..."
)
requirements = st.text_area(
    "Requirements (optional)",
    height=100,
    placeholder="Paste requirements if available..."
)

# Suspicious phrase checker
SUSPICIOUS = [
    "no experience needed", "work from home earn",
    "send bank details", "guaranteed income",
    "easy money", "whatsapp only", "no interview",
    "immediate joining", "pay registration fee",
    "earn per day", "part time earn", "no qualification"
]

def highlight_suspicious(text):
    flagged = []
    text_lower = text.lower()
    for phrase in SUSPICIOUS:
        if phrase in text_lower:
            flagged.append(phrase)
    return flagged

# Analyse button
if st.button("Analyse this job posting", type="primary"):
    if not title or not description:
        st.warning("Please enter at least the job title and description.")
    else:
        with st.spinner("Analysing..."):
            result = predict_job(title, description, requirements)

        st.markdown("---")

        # Main result
        fake_prob = float(result['fake_prob'])
        real_prob = float(result['real_prob'])

        if result['label'] == "FAKE":
            st.error(f"⚠️ LIKELY FAKE — {fake_prob}% confidence")
            st.markdown("This posting shows patterns commonly found in scam listings.")
        else:
            st.success(f"✅ LIKELY REAL — {real_prob}% confidence")
            st.markdown("This posting looks legitimate based on our analysis.")

        # Confidence bars
        st.markdown("#### Confidence breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Real probability", f"{real_prob}%")
            st.progress(real_prob / 100)
        with col_b:
            st.metric("Fake probability", f"{fake_prob}%")
            st.progress(fake_prob / 100)

        # Suspicious phrases
        flagged = highlight_suspicious(description + " " + title)
        if flagged:
            st.markdown("#### 🚩 Suspicious phrases found")
            cols = st.columns(3)
            for i, phrase in enumerate(flagged):
                with cols[i % 3]:
                    st.markdown(
                        f'<span style="background:#FCEBEB;color:#A32D2D;'
                        f'padding:4px 10px;border-radius:6px;'
                        f'font-size:13px;display:block;margin:3px">'
                        f'{phrase}</span>',
                        unsafe_allow_html=True
                    )
        else:
            st.markdown("#### Suspicious phrases")
            st.markdown("No obvious suspicious phrases detected.")

        # Red flags info
        st.markdown("---")
        st.markdown("#### Common red flags to watch for")
        st.info(
            "🔴 Requests for bank or personal details before hiring\n\n"
            "🔴 Unrealistic salary (Rs.50,000+/day for basic tasks)\n\n"
            "🔴 No company website or verifiable contact info\n\n"
            "🔴 WhatsApp-only communication\n\n"
            "🔴 Registration or training fee required before joining"
        )

st.markdown("---")
st.caption("Built with NLP + Scikit-learn + Streamlit | SDG 8 – Decent Work & Economy")
