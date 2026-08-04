import os    #operating system
import time
import pandas as pd
import streamlit as st
from google import genai
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu


def render_neon_chart(title, icon, fig):
    st.markdown(
        f"<div class='chart-card'><div class='neon-title'>{icon} {title}</div>",
        unsafe_allow_html=True,
    )
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


load_dotenv()
api_key = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("API_KEY")
)
client = genai.Client(api_key=api_key) if api_key else None

st.set_page_config(page_title="Heart Disease Prediction",page_icon="🫀")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #174b6b 0%, #0a1428 35%, #050b14 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07131f 0%, #11263c 100%);
        border-right: 1px solid rgba(125,211,252,0.18);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(56,189,248,0.22), rgba(14,116,144,0.35));
        border: 1px solid rgba(125,211,252,0.28);
        border-radius: 24px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        animation: floatNeon 5s ease-in-out infinite;
    }
    .hero-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, rgba(255,255,255,0.08), transparent 45%, rgba(255,255,255,0.08));
        pointer-events: none;
    }
    .hero-card h1 {
        color: #f8fafc;
        margin-bottom: 0.25rem;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .hero-card p {
        color: #dbeafe;
        margin-bottom: 0;
        font-size: 1rem;
    }
    .glass-panel {
        background: rgba(8, 20, 36, 0.78);
        border: 1px solid rgba(125,211,252,0.16);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        margin-bottom: 0.8rem;
    }
    .section-title {
        color: #e0f2fe;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .neon-title {
        color: #7dd3fc;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        text-shadow: 0 0 10px rgba(34,211,238,0.26);
    }
    .chart-card {
        background: rgba(7, 18, 31, 0.82);
        border: 1px solid rgba(34,211,238,0.22);
        border-radius: 18px;
        padding: 0.8rem 0.9rem 0.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 0 24px rgba(34,211,238,0.14);
        animation: glowPulse 2.7s ease-in-out infinite;
    }
    @keyframes floatNeon {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 18px rgba(34,211,238,0.12); }
        50% { box-shadow: 0 0 30px rgba(34,211,238,0.24); }
    }
    .stButton > button {
        background: linear-gradient(90deg, #22d3ee, #34d399);
        color: #052e16;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(34,211,238,0.24);
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(2, 8, 23, 0.95), rgba(15, 23, 42, 0.92));
        border: 1px solid rgba(34,211,238,0.24);
        border-radius: 16px;
        padding: 0.8rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.22);
        animation: pulseGlow 2.2s ease-in-out infinite;
    }
    .live-banner {
        background: linear-gradient(90deg, rgba(8, 145, 178, 0.24), rgba(16, 185, 129, 0.2));
        border: 1px solid rgba(34,211,238,0.28);
        border-radius: 16px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 0 22px rgba(34,211,238,0.18);
    }
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #22d3ee;
        box-shadow: 0 0 14px #22d3ee;
        margin-right: 0.45rem;
        animation: blink 1.2s infinite;
    }
    .control-room {
        background: linear-gradient(135deg, rgba(2, 8, 23, 0.86), rgba(10, 24, 42, 0.95));
        border: 1px solid rgba(34,211,238,0.2);
        border-radius: 20px;
        padding: 0.95rem;
        box-shadow: inset 0 0 30px rgba(34,211,238,0.08), 0 0 24px rgba(34,211,238,0.12);
    }
    .vital-card {
        background: linear-gradient(135deg, rgba(6, 17, 31, 0.96), rgba(12, 29, 49, 0.95));
        border: 1px solid rgba(34,211,238,0.2);
        border-radius: 16px;
        padding: 0.8rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 0 18px rgba(34,211,238,0.12);
    }
    .pulse-line {
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #f472b6);
        margin-top: 0.35rem;
        animation: pulseMove 1.6s linear infinite;
    }
    .hud-label {
        color: #7dd3fc;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .hud-value {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 800;
    }
    .alert-card {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.95), rgba(185, 28, 28, 0.9));
        border: 1px solid rgba(248, 113, 113, 0.45);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 0 24px rgba(248, 113, 113, 0.24);
        animation: alertPulse 1.2s infinite;
    }
    .pulse-signal {
        width: 100%;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #ef4444);
        margin-top: 0.45rem;
        animation: pulseMove 1.2s linear infinite;
    }
    .icu-panel {
        background: linear-gradient(135deg, rgba(4, 17, 31, 0.9), rgba(6, 30, 48, 0.95));
        border: 1px solid rgba(34,211,238,0.18);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: inset 0 0 28px rgba(34,211,238,0.08), 0 0 18px rgba(0,0,0,0.24);
    }
    .icu-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
    .waveform {
        height: 90px;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(2, 8, 23, 0.95), rgba(6, 30, 48, 0.95));
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(34,211,238,0.18);
        margin-top: 0.5rem;
    }
    .waveform::before {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(90deg, transparent 0 8px, rgba(34,211,238,0.08) 8px 16px);
        animation: scan 1.4s linear infinite;
    }
    .waveform::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, transparent 0%, rgba(248,113,113,0.18) 50%, transparent 100%);
        animation: wave 1.1s ease-in-out infinite;
        clip-path: polygon(0 60%, 8% 57%, 15% 67%, 25% 40%, 35% 72%, 45% 30%, 56% 73%, 66% 50%, 78% 64%, 88% 45%, 100% 70%, 100% 100%, 0 100%);
    }
    .monitor-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.6rem;
        margin-top: 0.6rem;
    }
    .monitor-panel {
        background: linear-gradient(135deg, rgba(3, 12, 24, 0.97), rgba(8, 20, 34, 0.96));
        border: 1px solid rgba(34,211,238,0.22);
        border-radius: 16px;
        padding: 0.8rem;
        box-shadow: 0 0 18px rgba(34,211,238,0.14), 0 0 22px rgba(248,113,113,0.14);
        animation: panelFloat 4s ease-in-out infinite;
    }
    .ecg-line {
        height: 120px;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(2, 8, 23, 0.95), rgba(6, 30, 48, 0.95));
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(34,211,238,0.16);
        margin-top: 0.4rem;
    }
    .ecg-line::before {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(90deg, transparent 0 10px, rgba(34,211,238,0.07) 10px 20px);
        opacity: 0.5;
    }
    .ecg-line::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(248,113,113,0.12) 40%, rgba(34,211,238,0.18) 60%, transparent 100%);
        animation: ecgFlow 1.3s ease-in-out infinite;
        clip-path: polygon(0 70%, 8% 70%, 12% 60%, 20% 58%, 25% 72%, 33% 42%, 42% 78%, 51% 50%, 58% 55%, 67% 32%, 76% 78%, 84% 48%, 90% 69%, 100% 65%, 100% 100%, 0 100%);
    }
    .critical-care {
        position: fixed;
        inset: 0;
        background: radial-gradient(circle at center, rgba(3, 8, 20, 0.98), rgba(0, 0, 0, 0.98));
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        animation: cinematicFade 1.2s ease-in-out;
    }
    .critical-care .inner {
        border: 1px solid rgba(248,113,113,0.35);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(3, 7, 18, 0.98));
        box-shadow: 0 0 60px rgba(248,113,113,0.34);
        padding: 1rem;
        width: 100%;
        max-width: 1100px;
        animation: panelSpin 8s linear infinite;
        position: relative;
        overflow: hidden;
    }
    .pulse-screen {
        background: linear-gradient(135deg, rgba(2, 8, 23, 0.98), rgba(7, 18, 31, 0.98));
        border: 1px solid rgba(248,113,113,0.24);
        border-radius: 18px;
        padding: 0.8rem;
        min-height: 260px;
        box-shadow: inset 0 0 24px rgba(248,113,113,0.12), 0 0 22px rgba(248,113,113,0.18);
    }
    .pulse-graph {
        height: 140px;
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(2, 8, 23, 0.98), rgba(6, 30, 48, 0.98));
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(248,113,113,0.2);
    }
    .pulse-graph::before {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(90deg, transparent 0 10px, rgba(248,113,113,0.06) 10px 20px);
    }
    .pulse-graph::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(248,113,113,0.18) 40%, rgba(34,211,238,0.24) 60%, transparent 100%);
        animation: pulseBeat 1s infinite;
        clip-path: polygon(0 68%, 8% 62%, 16% 74%, 26% 46%, 34% 82%, 44% 38%, 54% 74%, 64% 54%, 74% 80%, 84% 38%, 92% 68%, 100% 58%, 100% 100%, 0 100%);
    }
    .equipment-panel {
        background: linear-gradient(135deg, rgba(8, 20, 36, 0.95), rgba(4, 12, 24, 0.98));
        border: 1px solid rgba(34,211,238,0.2);
        border-radius: 16px;
        padding: 0.7rem;
        min-height: 120px;
        box-shadow: inset 0 0 18px rgba(34,211,238,0.08);
    }
    .code-blue {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.95), rgba(239, 68, 68, 0.95));
        color: white;
        border-radius: 18px;
        padding: 1rem;
        text-align: center;
        font-weight: 900;
        box-shadow: 0 0 40px rgba(248,113,113,0.45);
        animation: codeBlueFlash 0.8s infinite;
    }
    .overlay-glow {
        position: absolute;
        inset: 0;
        background: radial-gradient(circle, rgba(248,113,113,0.16), transparent 70%);
        pointer-events: none;
        animation: glowPulse 1.4s ease-in-out infinite;
    }
    .monitor-card {
        background: linear-gradient(135deg, rgba(2, 8, 23, 0.96), rgba(13, 23, 42, 0.96));
        border: 1px solid rgba(34,211,238,0.24);
        border-radius: 18px;
        padding: 0.8rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 0 20px rgba(34,211,238,0.14);
    }
    .monitor-value {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
    }
    .monitor-label {
        color: #7dd3fc;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .emergency-overlay {
        position: fixed;
        inset: 0;
        background: rgba(10, 4, 4, 0.95);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
    }
    .emergency-box {
        border: 2px solid #ef4444;
        border-radius: 24px;
        background: radial-gradient(circle at center, rgba(127,29,29,0.9), rgba(17, 24, 39, 0.96));
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.35);
        max-width: 560px;
        width: 100%;
        animation: flashRed 0.8s infinite;
    }
    .emergency-title {
        color: #fef2f2;
        font-size: 1.5rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }
    .emergency-subtitle {
        color: #fecaca;
        font-size: 1rem;
    }
    @keyframes alertPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.01); }
    }
    @keyframes flashRed {
        0%, 100% { background: radial-gradient(circle at center, rgba(127,29,29,0.9), rgba(17, 24, 39, 0.96)); }
        50% { background: radial-gradient(circle at center, rgba(239,68,68,0.96), rgba(127,29,29,0.9)); }
    }
    @keyframes scan {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    @keyframes wave {
        0%, 100% { opacity: 0.25; transform: translateY(0); }
        50% { opacity: 0.8; transform: translateY(-4px); }
    }
    @keyframes ecgFlow {
        0%, 100% { opacity: 0.35; transform: translateX(-5px); }
        50% { opacity: 0.8; transform: translateX(5px); }
    }
    @keyframes panelFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-3px); }
    }
    @keyframes panelSpin {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(0.35deg); }
    }
    @keyframes pulseBeat {
        0%, 100% { opacity: 0.25; transform: translateY(0); }
        50% { opacity: 0.95; transform: translateY(-4px); }
    }
    @keyframes codeBlueFlash {
        0%, 100% { box-shadow: 0 0 16px rgba(248,113,113,0.25); }
        50% { box-shadow: 0 0 35px rgba(248,113,113,0.45); }
    }
    @keyframes cinematicFade {
        from { opacity: 0.2; }
        to { opacity: 1; }
    }
    @keyframes glowPulse {
        0%, 100% { opacity: 0.35; }
        50% { opacity: 0.75; }
    }
    @keyframes pulseMove {
        0% { transform: scaleX(0.9); opacity: 0.75; }
        50% { transform: scaleX(1.05); opacity: 1; }
        100% { transform: scaleX(0.9); opacity: 0.75; }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 8px 20px rgba(0,0,0,0.22); }
        50% { box-shadow: 0 0 22px rgba(34,211,238,0.24); }
    }
    @keyframes blink {
        0%, 100% { opacity: 0.5; transform: scale(0.95); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }
    .stTabs [role="tablist"] {
        gap: 0.35rem;
    }
    .stTabs [role="tab"] {
        border-radius: 999px;
        padding: 0.45rem 0.8rem;
        color: #dbeafe;
        border: 1px solid rgba(125,211,252,0.16);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #22d3ee, #818cf8);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>🫀 Heart Disease Prediction Dashboard</h1>
        <p>Explore model performance, visualize medical insights, and get a clear health overview in one place.</p>
    </div>
    <div class="live-banner">
        <span class="live-dot"></span><strong>Live Health Status:</strong> Monitoring system active · AI insights ready · Neon diagnostics online
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<h1 style='text-align:center; color:#f8fafc'>🫀</h1>",unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:#f8fafc'>Heart Disease</h2>",unsafe_allow_html=True)
    st.caption("Dashboard")
    select = option_menu(menu_title="Navigation", options=["Summery","Visualization","Prediction"],icons=["table","bar-chart","cpu"],default_index=0)
    st.info("Shashi's Heart Prediction System")
#1st part
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "Dataset", "heart.csv")

if not os.path.exists(dataset_path):
    st.error(f"Dataset file not found: {dataset_path}")
    st.stop()

df = pd.read_csv(dataset_path)

x = df.drop("target",axis=1)
y = df["target"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

tree_model = DecisionTreeClassifier()
log_model = LogisticRegression()
forest_model = RandomForestClassifier()

tree_model.fit(x_train, y_train)
log_model.fit(x_train_scaler,y_train)
forest_model.fit(x_train,y_train)

tree_pred = tree_model.predict(x_test)
log_pred = log_model.predict(x_test_scaler)
forest_pred = forest_model.predict(x_test)

tree_acc = accuracy_score(y_test, tree_pred)
log_acc = accuracy_score(y_test,log_pred)
forest_acc = accuracy_score(y_test,forest_pred)

model_result = pd.DataFrame({
    "Model":["Decision Tree","Random Forest","Logistic Regression"],
    "Accuracy":[tree_acc,log_acc,forest_acc]
})

model_result = model_result.sort_values(by="Accuracy",ascending=False)

best_model_name = model_result.iloc[0]["Model"]
best_accuracy = model_result.iloc[0]["Accuracy"]

if select=="Summery":

    st.markdown("<div class='control-room'><div class='section-title'>🧪 Clinical Evaluation Console</div></div>", unsafe_allow_html=True)
    st.subheader("📊 Model Comparison")
    st.dataframe(model_result,use_container_width=True)

    st.markdown(
        """
        <div class='vital-card'>
            <div class='hud-label'>Critical Care Summary</div>
            <div class='hud-value'>Best model: {}</div>
            <div class='pulse-line'></div>
        </div>
        """.format(best_model_name),
        unsafe_allow_html=True,
    )

    st.success(f"Best Model : {best_model_name}")
    st.info(f"Best Accuracy : {best_accuracy*100:.2f}")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("<div class='vital-card'><div class='hud-label'>Logistic</div><div class='hud-value'>{}</div><div class='pulse-line'></div></div>".format(f"{log_acc*100:.2f} %"), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='vital-card'><div class='hud-label'>Decision Tree</div><div class='hud-value'>{}</div><div class='pulse-line'></div></div>".format(f"{tree_acc*100:.2f} %"), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='vital-card'><div class='hud-label'>Random Forest</div><div class='hud-value'>{}</div><div class='pulse-line'></div></div>".format(f"{forest_acc*100:.2f} %"), unsafe_allow_html=True)



#2nd part
if select== "Visualization":

    st.markdown("<div class='control-room'><div class='section-title'>🛰️ Hospital Control Room</div></div>", unsafe_allow_html=True)
    st.subheader("📈 Visualization Hub")
    tab1,tab2,tab3 = st.tabs(["Distribution","Relationship","Heatmap"])

    with tab1:
        st.subheader("Distribution Graph")
        c1,c2 = st.columns(2)

        with c1:
            fig,ax = plt.subplots(figsize=(4,3), facecolor="#07131f")
            target = df["target"].value_counts()
            ax.bar(["Healthy","Disease"],target.values, color=["#22d3ee", "#f472b6"])
            ax.set_facecolor("#07131f")
            ax.set_title("Heart Disease Distribution", color="#e0f2fe")
            ax.tick_params(colors="#e0f2fe")
            for spine in ax.spines.values():
                spine.set_color("#38bdf8")
            placeholder = st.empty()
            for _ in range(3):
                placeholder.pyplot(fig, use_container_width=True)
                time.sleep(0.12)

        with c2:
            fig,ax = plt.subplots(figsize=(4,3), facecolor="#07131f")
            gender = df["gender"].value_counts()
            ax.bar(["Male","Female"],gender.values, color=["#34d399", "#818cf8"])
            ax.set_facecolor("#07131f")
            ax.set_title("Gender Distribution", color="#e0f2fe")
            ax.tick_params(colors="#e0f2fe")
            for spine in ax.spines.values():
                spine.set_color("#38bdf8")
            render_neon_chart("Gender Distribution", "👩‍⚕️", fig)


        c3,c4 = st.columns(2)
        with c3:
            fig,ax= plt.subplots(figsize=(4,3), facecolor="#07131f")
            ax.hist(df["age"],bins=25,color="#67e8f9")
            ax.set_facecolor("#07131f")
            ax.set_xlabel("Age", color="#e0f2fe")
            ax.set_ylabel("Count", color="#e0f2fe")
            ax.tick_params(colors="#e0f2fe")
            for spine in ax.spines.values():
                spine.set_color("#38bdf8")
            render_neon_chart("Age Distribution", "🧓", fig)

        with c4:
            fig,ax= plt.subplots(figsize=(4,3), facecolor="#07131f")
            ax.hist(df["age"],bins=25,color="#f0abfc")
            ax.set_facecolor("#07131f")
            ax.set_xlabel("Age", color="#e0f2fe")
            ax.set_ylabel("Count", color="#e0f2fe")
            ax.tick_params(colors="#e0f2fe")
            for spine in ax.spines.values():
                spine.set_color("#38bdf8")
            render_neon_chart("Age Distribution", "📈", fig)



    with tab2:
        st.subheader("Relationship Graph")

        c1,c2 = st.columns(2)
        with c1:
            fig,ax = plt.subplots(figsize=(4,3), facecolor="#07131f")
            ax.scatter(df["age"],df["chol"],alpha=0.8,c=df["age"],cmap="cool",s=7)
            ax.set_facecolor("#07131f")
            ax.set_xlabel("Age", color="#e0f2fe")
            ax.set_ylabel("Cholesterol", color="#e0f2fe")
            ax.tick_params(colors="#e0f2fe")
            for spine in ax.spines.values():
                spine.set_color("#38bdf8")
            render_neon_chart("Age vs Cholesterol", "🫀", fig)

        with c2:
                fig,ax = plt.subplots(figsize=(4,3), facecolor="#07131f")
                ax.scatter(df["trestbps"],df["thalachh"],alpha=0.8,c=df["trestbps"],cmap="ocean",s=15)
                ax.set_facecolor("#07131f")
                ax.set_xlabel("Blood Pressure", color="#e0f2fe")
                ax.set_ylabel("Heart Rate", color="#e0f2fe")
                ax.tick_params(colors="#e0f2fe")
                for spine in ax.spines.values():
                    spine.set_color("#38bdf8")
                render_neon_chart("Blood Pressure vs Heart Rate", "💓", fig)



    with tab3:
        fig,ax = plt.subplots(figsize=(8,5), facecolor="#07131f")
        heat = ax.imshow(df.corr(),cmap="coolwarm")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns,rotation=90,fontsize=10,color="#e0f2fe")
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns,fontsize=10,color="#e0f2fe")
        ax.set_facecolor("#07131f")
        for spine in ax.spines.values():
            spine.set_color("#38bdf8")
        plt.colorbar(heat, ax=ax)
        render_neon_chart("Correlation Heatmap", "🧠", fig)







#3rd part
if select =="Prediction":

    st.markdown("<div class='control-room'><div class='section-title'>🩺 Patient Risk Monitoring Panel</div></div>", unsafe_allow_html=True)
    st.subheader("🩺 Heart Disease Prediction")
    c1,c2 = st.columns(2)
    with c1:
        age = st.number_input("Enter your age:",18,100,20)
        gender = st.selectbox("Gender:",[0,1],format_func=lambda x:"Female" if x==0 else "Male")
        cp = st.selectbox("Chest Pain",[0,1,2,3])
        trestbps = st.number_input("Resting blood pressure",80,250,120)
        chol = st.number_input("Cholesterol",100,600,200)
        fbs = st.selectbox("Fasting Blood Sugar",[0,1])
        restecg = st.selectbox("Resting Electrocardiographic :",[0,1,2])

    with c2:
        thalachh = st.number_input("Maximum Heart Rate",60,220,150)
        exang = st.selectbox("Exersice:",[0,1],format_func=lambda x: "No" if x==0 else "Yes")
        oldpeak = st.number_input("OLd peak :",0.0,10.0,1.0)
        slope = st.selectbox("Slope:",[0,1,2])
        ca = st.selectbox("Major vessel:",[0,1,2,3,4])
        thal = st.selectbox("Thalassemia:",[0,1,2,3])

        st.divider()

        if st.button("Predict",use_container_width=True):
            patient = pd.DataFrame([{
                "age":age,
                "gender":gender,
                "cp":cp,
                "trestbps":trestbps,
                "chol":chol,
                "fbs":fbs,
                "restecg":restecg,
                "thalachh":thalachh,
                "exang":exang,
                "oldpeak":oldpeak,
                "slope":slope,
                "ca":ca,
                "thal":thal
            }])
            if best_model_name=="Logistic Regression":
                result =log_model.predict(scaler.transform(patient))
            elif best_model_name=="Decision Tree":
                result=tree_model.predict(patient)
            else:
                result=forest_model.predict(patient)

            st.markdown("<div class='icu-panel'><div class='icu-title'>🧬 ICU Risk Assessment</div></div>", unsafe_allow_html=True)
            st.subheader("Prediction Result")
            if result[0] == 1:
                st.markdown(
                    """
                    <div class='emergency-overlay'>
                        <div class='emergency-box'>
                            <div class='emergency-title'>🚨 EMERGENCY ALERT</div>
                            <div class='emergency-subtitle'>High-risk cardiovascular condition detected. Immediate medical consultation is strongly advised.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class='alert-card'>
                        <div class='hud-label'>RED ALERT</div>
                        <div class='hud-value'>High risk of Heart Disease</div>
                        <div class='pulse-signal'></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='monitor-card'><div class='monitor-label'>Heart Rate</div><div class='monitor-value'>98 bpm</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown("<div class='monitor-card'><div class='monitor-label'>SpO2</div><div class='monitor-value'>91%</div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown("<div class='monitor-card'><div class='monitor-label'>Status</div><div class='monitor-value'>Critical</div></div>", unsafe_allow_html=True)
                st.markdown("<div class='monitor-grid'><div class='monitor-panel'><div class='monitor-label'>ECG Waveform</div><div class='ecg-line'></div></div><div class='monitor-panel'><div class='monitor-label'>Respiration</div><div class='monitor-value'>24/min</div></div><div class='monitor-panel'><div class='monitor-label'>Blood Pressure</div><div class='monitor-value'>145/92</div></div></div>", unsafe_allow_html=True)

                cp_map = {
                    0: "Typical Angina",
                    1: "Atypical Angina",
                    2: "Non-Anginal Pain",
                    3: "Asymptomatic"
                }
                restecg_map = {
                    0: "Normal",
                    1: "AT-T Wave Abnormality",
                    2: "Left Ventricular Hypertrophy"
                }
                slope_map = {
                    0: "Upsloping",
                    1: "Flat",
                    2: "Downsloping"
                }
                thal_map = {
                    0: "Normal",
                    1: "Fixed Defect",
                    2: "Reversable Defect",
                    3: "Unknown"
                }
                prompt = f"""
                Your are a Experieence Cardiologist.
                A Paiteint Heart Disease prediction model classfication this patient as HIGH Risk.
                Patient Details
                Age:{age}
                Gender:{"Male" if gender==1 else "Female"}
                Chest Pain Type:{cp_map[cp]}
                Resting Blood Pressure:{trestbps}
                Cholestrol:{chol}
                Fasting Blood Suger:{"High" if fbs==1 else "Normal"}
                Resting Electrocardiographic:{restecg_map[restecg]}
                Maximum Heart Rate: {thalachh}
                Exercise Engina: {"Yes" if exang ==1 else "NO"}
                Old Peak :{oldpeak}
                Slope :{slope_map[slope]}
                Major vessal:{ca}
                Thalassemia : {thal_map[thal]}

                Give Response in this formet.
                Risk Analysis
                Possible Reasons
                Life Style Advice
                Diet Instructions 
                Exercise
                Doctor Recmondation
                Keep Response under 500 word.
                also give patient summery in proper ptofessional Formet.

                """

                st.subheader("AI Health Suggestion")

            if client is None:
                st.warning("AI suggestions are disabled because no Gemini API key is configured. Add GEMINI_API_KEY to your environment or .env file to enable them.")
                st.info("""
                - Contact with Doctor
                - Monitor Blodd Pressure 
                - Reduce Cholestrol
                """)
            else:
                try:
                    with st.spinner("Genrating AI report........."):
                        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
                    st.markdown(response.text)
                except Exception:
                    st.warning("AI Server is Busy. this is by default suggestion")
                    st.info("""
                    - Contact with Doctor
                    - Monitor Blodd Pressure 
                    - Reduce Cholestrol
                    """)


        else:
            st.success("Low risk  of Heart disease")
            st.subheader ("AI Health Suggestion: ")
            st.info("""
            - Continue your daily routine 
            - Regular Exercise 
            - Balence Your Diet 
            - Complete your 8 hour sleep cycle 
            """)

st.markdown("<p style='text-align:center'>Heart Prediction System Developed By Shashi Kumar Singh</p>",unsafe_allow_html=True)