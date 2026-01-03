import streamlit as st
import pandas as pd
import numpy as np
import pickle

#---SAYFA AYARLARI
st.set_page_config(
    page_title="Spotify Churn AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TASARIM İÇİN ÖZEL CSS (SPOTIFY YEŞİLİ VE KOYU TEMA) ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 30px;
        height: 100%;
        width: 100%;
        display: flex;
        justify-content: center;
        font-size: 20px;
        font-weight: bold;
        border: none;
        box-shadow: 0px 4px 15px rgba(29,185,84,0.4);
    }
    .stButton>button:hover {
        background-color: #1ed760;
        color: white;
        transform: scale(1.04);
    }
    h1 {
        color: #1DB954;
    }
    h3 {
        color: #FFFFFF;
    }
    div[data-testid="stMetric"] {
        background-color: #181818;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. MODELİ YÜKLEME ---
@st.cache_resource
def load_data():
    with open('churn_model_final.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    data = load_data()
    model = data["model"]
    scaler = data["scaler"]
    feature_names = data["features"]
except FileNotFoundError:
    st.error("Model not found!")
    st.stop()

# --- 2. SAYFA TASARIMI ---
col1, col2, col3 = st.columns([1,6,1])

with col2:
    # Yan yana (Logo ve Yazı) durması için alt sütunlar
    head_c1, head_c2 = st.columns([1, 2])
    with head_c1:
        # Spotify Logosu (Web'den çekiyoruz)
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=100)
    with head_c2:
        st.title("Spotify Churn Predictor")
        st.caption("AI and Feature Engineering Powered Customer Churn Analysis")

with col2:
    st.write("---")
st.write(""" 
This application predicts the churn probability analysing Spotify user data. Enter user information on the menu.
""")



# --- 3. KULLANICI GİRDİLERİ (INPUTS) ---

st.subheader("👤 User Profile and Behaviour")

# Alanı ikiye bölüyoruz (Sol: Demografik, Sağ: Kullanım)
left_col, right_col = st.columns(2)

with left_col:
    st.info("📊 Demographic Information")

    c1, c2 = st.columns(2)
    age = c1.slider("Age", 15, 80, 25)
    gender = c2.selectbox("Gender", ["Female", "Male", "Other"])
    subscription = st.selectbox("Subscription Type", ["Free", "Premium", "Student", "Family"])
    country = st.selectbox("Country", ["TR", 'CA', 'DE', 'AU', 'US', 'UK', 'IN', 'FR', 'PK', "Other"])
    device = st.selectbox("Device Type", ["Mobile", "Desktop", "Web"])

with right_col:
    st.success("🎶 Listening Habits")
    listening_time = st.slider("Listening Time per Day (min)", 0, 500, 120,
                               help="The average number of minutes per day that the user listens to music.")
    songs_played = st.slider("Songs Played per Day", 0, 200, 20)
    skip_rate = st.slider("Skip Rate (0.0 - 1.0)", 0.0, 1.0, 0.2, 0.01,
                          help="1.0 means that you skips each song.")
    c3, c4 = st.columns(2)
    offline_listening = c3.number_input("Offline Listening", 0, 50, 5)

    # If subscription is not Free, Ads listened set to 0.
    if subscription == "Free":
        ads_disabled = False
        ads_value = 15
    else:
        # Premium/Family/Student ise 0'a sabitlenir ve kilitlenir
        ads_disabled = True
        ads_value = 0

    ads_listened_weekly = c4.number_input(
        "Ads Listened Weekly",
        value=ads_value,
        disabled=ads_disabled,
        help="Premium paketlerde reklam dinlenmediği için 0 olarak sabitlenmiştir."
    )

# --- 4. VERİYİ MODEL FORMATINA GETİRME ---
avg_song_duration = listening_time / (songs_played + 1)
ads_intensity = ads_listened_weekly / (listening_time + 1)
dissatisfaction_score = skip_rate * songs_played
ads_listened_log = np.log1p(ads_listened_weekly)

input_data = {
    'age': age,
    'listening_time': listening_time,
    'songs_played_per_day': songs_played,
    'skip_rate': skip_rate,
    'offline_listening': offline_listening,
    'ads_listened_log': ads_listened_log,
    'avg_song_duration': avg_song_duration,
    'ads_intensity': ads_intensity,
    'dissatisfaction_score': dissatisfaction_score
}

# One-Hot Encoding Mantığı
input_df = pd.DataFrame(0, index=[0], columns=feature_names)
for col in input_data:
    if col in input_df.columns:
        input_df[col] = input_data[col]

# Kategorik değerleri "1" yapalım (Mapping)
# Örnek: Kullanıcı 'Male' seçtiyse 'gender_Male' sütununu 1 yap
if f"gender_{gender}" in input_df.columns: input_df[f"gender_{gender}"] = 1
if f"subscription_type_{subscription}" in input_df.columns:input_df[f"subscription_type_{subscription}"] = 1
if f"device_type_{device}" in input_df.columns:input_df[f"device_type_{device}"] = 1
if f"country_{country}" in input_df.columns:input_df[f"country_{country}"] = 1

# --- 5. TAHMİN ---
b_col1, b_col2, b_col3 = st.columns([10,2,9])

with b_col2:
    predict_btn = st.button("PREDICT")

if predict_btn:
    with st.spinner('AI Analysis is starting...'):
        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[0][1]

    st.write("---")
    st.subheader('Result of Analysis')

    res_col1, res_col2, res_col3 = st.columns([1,2,1])

    with res_col2:
        if probability >= 0.30:
            # RİSKLİ DURUM
            st.error("⚠️ **HIGH RISK**")
            st.metric(label="Churn Probability", value=f"%{probability * 100:.1f}", delta="-Critical",delta_color="inverse")
            st.progress(int(probability * 100))
            st.write("Suggest: This user should be offered a special discount.")
        else:
            # GÜVENLİ DURUM
            st.success("✅ **LOYAL**")
            st.metric(label="Churn Probability", value=f"%{probability * 100:.1f}", delta="+Safe")
            st.progress(int(probability * 100))
            st.write("Status: This user seems satisfied.")

# Alt bilgi
st.write("---")
st.caption("Developed by Ozan Serhan Denizhan & Nisa Kızılkaya & Miray Şenay | COE305 Machine Learning Project")
