import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedTriage AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if not (BASE_DIR / "models").exists():
    if (BASE_DIR.parent / "models").exists():
        BASE_DIR = BASE_DIR.parent

MODEL_PATH = BASE_DIR / "models" / "triage_model_bundle.pkl"
DATA_PATH = BASE_DIR / "data" / "triage_training_merged.csv"
PATIENT_FILE = BASE_DIR / "data" / "patients.csv"


# ============================================================
# DESIGN
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: #F4F7FB;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0B5CAD;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    color: white !important;
    text-align: left !important;
    font-size: 15px !important;
}

[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,0.15) !important;
}

/* Main headings */
h1, h2, h3 {
    color: #102A43 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    border-radius: 8px;
}

/* Select boxes */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #E2E8F0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ============================================================
# NAVIGATION
# ============================================================

def navigate(page):
    st.session_state.page = page


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


model_bundle = load_model()


# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data():

    if not DATA_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return pd.DataFrame()


training_df = load_training_data()


# ============================================================
# PATIENT FUNCTIONS
# ============================================================

def load_patients():

    if not PATIENT_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(PATIENT_FILE)
    except Exception:
        return pd.DataFrame()


def save_patient(patient):

    PATIENT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    patients = load_patients()

    new_patient = pd.DataFrame([patient])

    if patients.empty:
        patients = new_patient
    else:
        patients = pd.concat(
            [patients, new_patient],
            ignore_index=True
        )

    patients.to_csv(
        PATIENT_FILE,
        index=False
    )


patients_df = load_patients()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🏥 MedTriage AI")

    st.caption("Healthcare Management")

    st.divider()

    st.markdown("### Navigation")

    if st.button(
        "🏠  Dashboard",
        use_container_width=True
    ):
        navigate("Dashboard")
        st.rerun()

    if st.button(
        "👤  New Patient",
        use_container_width=True
    ):
        navigate("New Patient")
        st.rerun()

    if st.button(
        "🩺  Triage Assessment",
        use_container_width=True
    ):
        navigate("Triage Assessment")
        st.rerun()

    if st.button(
        "📋  Patient Queue",
        use_container_width=True
    ):
        navigate("Patient Queue")
        st.rerun()

    if st.button(
        "📊  Reports & Analytics",
        use_container_width=True
    ):
        navigate("Reports & Analytics")
        st.rerun()

    if st.button(
        "⚙️  Settings",
        use_container_width=True
    ):
        navigate("Settings")
        st.rerun()

    st.divider()

    st.caption("MedTriage AI")
    st.caption("AI-Powered Triage Management")


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    # Header
    st.title("🏥 MedTriage AI")

    st.write(
        "Welcome back. Manage patients and perform "
        "AI-assisted triage from one place."
    )

    st.divider()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_patients = len(patients_df)

    today = datetime.now().strftime("%Y-%m-%d")

    if (
        not patients_df.empty
        and "date" in patients_df.columns
    ):
        today_patients = len(
            patients_df[
                patients_df["date"]
                .astype(str)
                .str.startswith(today)
            ]
        )
    else:
        today_patients = 0

    if (
        not patients_df.empty
        and "urgency_level" in patients_df.columns
    ):
        emergency_cases = len(
            patients_df[
                patients_df["urgency_level"]
                .astype(str)
                .str.lower()
                .str.contains("emergency", na=False)
            ]
        )
    else:
        emergency_cases = 0

    if (
        not patients_df.empty
        and "status" in patients_df.columns
    ):
        pending_cases = len(
            patients_df[
                patients_df["status"]
                .astype(str)
                .str.lower()
                == "pending"
            ]
        )
    else:
        pending_cases = 0

    # Native Streamlit metric cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Patients",
            total_patients
        )

    with c2:
        st.metric(
            "Today's Patients",
            today_patients
        )

    with c3:
        st.metric(
            "Emergency Cases",
            emergency_cases
        )

    with c4:
        st.metric(
            "Pending Queue",
            pending_cases
        )

    st.write("")

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.subheader("Quick Actions")

    q1, q2, q3 = st.columns(3)

    with q1:

        if st.button(
            "➕ Register New Patient",
            use_container_width=True
        ):
            navigate("New Patient")
            st.rerun()

    with q2:

        if st.button(
            "🩺 Start AI Triage",
            use_container_width=True
        ):
            navigate("Triage Assessment")
            st.rerun()

    with q3:

        if st.button(
            "📋 View Patient Queue",
            use_container_width=True
        ):
            navigate("Patient Queue")
            st.rerun()

    st.write("")

    # --------------------------------------------------------
    # DASHBOARD CONTENT
    # --------------------------------------------------------

    left, right = st.columns([2, 1])

    with left:

        st.subheader("Recent Patients")

        if patients_df.empty:

            st.info(
                "No patients registered yet."
            )

            if st.button(
                "Register First Patient"
            ):
                navigate("New Patient")
                st.rerun()

        else:

            columns = [
                "patient_id",
                "name",
                "age",
                "sex",
                "chief_complaint",
                "urgency_level",
                "status"
            ]

            available = [
                col
                for col in columns
                if col in patients_df.columns
            ]

            recent = patients_df.tail(8)

            st.dataframe(
                recent[available],
                use_container_width=True,
                hide_index=True
            )

    with right:

        st.subheader("System Status")

        if model_bundle is not None:

            st.success(
                "🟢 AI Model Online"
            )

        else:

            st.error(
                "🔴 AI Model Offline"
            )

        if not training_df.empty:

            st.info(
                f"Dataset: {training_df.shape[0]:,} records"
            )

        else:

            st.warning(
                "Training dataset unavailable"
            )

        st.info(
            "Educational prototype"
        )

    st.write("")

    # --------------------------------------------------------
    # URGENCY OVERVIEW
    # --------------------------------------------------------

    if (
        not patients_df.empty
        and "urgency_level" in patients_df.columns
    ):

        st.subheader("Triage Overview")

        urgency_data = (
            patients_df[
                "urgency_level"
            ]
            .value_counts()
        )

        st.bar_chart(
            urgency_data
        )

    st.caption(
        "For educational and demonstration purposes only. "
        "Not intended for real clinical decision-making."
    )


# ============================================================
# NEW PATIENT
# ============================================================

elif st.session_state.page == "New Patient":

    st.title("👤 New Patient")

    st.write(
        "Register a patient before starting the triage assessment."
    )

    st.divider()

    with st.form("new_patient_form"):

        col1, col2 = st.columns(2)

        with col1:

            name = st.text_input(
                "Patient Name"
            )

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=25
            )

            sex = st.selectbox(
                "Sex",
                [
                    "Male",
                    "Female",
                    "Other"
                ]
            )

        with col2:

            phone = st.text_input(
                "Phone Number"
            )

            arrival_mode = st.selectbox(
                "Arrival Mode",
                [
                    "Walk-in",
                    "Ambulance",
                    "Private Vehicle",
                    "Public Transport"
                ]
            )

            insurance_type = st.selectbox(
                "Insurance Type",
                [
                    "Private",
                    "Government",
                    "Self Pay",
                    "Other"
                ]
            )

        chief_complaint = st.text_area(
            "Chief Complaint",
            placeholder="Describe the patient's main complaint..."
        )

        submitted = st.form_submit_button(
            "Register Patient",
            use_container_width=True
        )

        if submitted:

            if not name.strip():

                st.error(
                    "Please enter the patient's name."
                )

            else:

                patient_id = (
                    "PT"
                    + datetime.now().strftime(
                        "%Y%m%d%H%M%S"
                    )
                )

                patient = {
                    "patient_id": patient_id,
                    "name": name,
                    "age": age,
                    "sex": sex,
                    "phone": phone,
                    "arrival_mode": arrival_mode,
                    "insurance_type": insurance_type,
                    "chief_complaint": chief_complaint,
                    "status": "Pending",
                    "date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }

                save_patient(patient)

                st.session_state.patient_data = patient

                st.success(
                    f"Patient registered successfully! "
                    f"Patient ID: {patient_id}"
                )


# ============================================================
# TRIAGE ASSESSMENT
# ============================================================

elif st.session_state.page == "Triage Assessment":

    st.title("🩺 AI Triage Assessment")

    st.write(
        "Enter patient information to estimate the urgency level."
    )

    st.divider()

    if model_bundle is None:

        st.error(
            "Machine learning model not found."
        )

        st.info(
            f"Expected location: {MODEL_PATH}"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            patient_name = st.text_input(
                "Patient Name",
                value=st.session_state.patient_data.get(
                    "name",
                    ""
                )
            )

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=25
            )

            sex = st.selectbox(
                "Sex",
                [
                    "Male",
                    "Female",
                    "Other"
                ]
            )

            arrival_mode = st.selectbox(
                "Arrival Mode",
                [
                    "Walk-in",
                    "Ambulance",
                    "Private Vehicle",
                    "Public Transport"
                ]
            )

            chief_complaint = st.text_input(
                "Chief Complaint"
            )

            pain_location = st.selectbox(
                "Pain Location",
                [
                    "None",
                    "Chest",
                    "Abdomen",
                    "Head",
                    "Back",
                    "Arm",
                    "Leg",
                    "Other"
                ]
            )

        with col2:

            mental_status = st.selectbox(
                "Mental Status",
                [
                    "Alert",
                    "Confused",
                    "Drowsy",
                    "Unresponsive"
                ]
            )

            pain_score = st.slider(
                "Pain Score",
                0,
                10,
                0
            )

            heart_rate = st.number_input(
                "Heart Rate",
                min_value=30,
                max_value=220,
                value=80
            )

            systolic_bp = st.number_input(
                "Systolic Blood Pressure",
                min_value=50,
                max_value=250,
                value=120
            )

            respiratory_rate = st.number_input(
                "Respiratory Rate",
                min_value=5,
                max_value=60,
                value=16
            )

            oxygen_saturation = st.number_input(
                "Oxygen Saturation (%)",
                min_value=50,
                max_value=100,
                value=98
            )

        st.write("")

        if st.button(
            "🔍 Predict Patient Urgency",
            use_container_width=True
        ):

            try:

                if isinstance(
                    model_bundle,
                    dict
                ):

                    preprocessor = (
                        model_bundle.get(
                            "preprocessor"
                        )
                    )

                    model = (
                        model_bundle.get(
                            "model"
                        )
                    )

                else:

                    preprocessor = None
                    model = model_bundle

                input_data = {
                    "age": age,
                    "sex": sex,
                    "arrival_mode": arrival_mode,
                    "chief_complaint_raw": chief_complaint,
                    "pain_location": pain_location,
                    "mental_status_triage": mental_status,
                    "pain_score": pain_score,
                    "heart_rate": heart_rate,
                    "systolic_bp": systolic_bp,
                    "respiratory_rate": respiratory_rate,
                    "oxygen_saturation": oxygen_saturation
                }

                input_df = pd.DataFrame(
                    [input_data]
                )

                if not training_df.empty:

                    excluded = [
                        "triage_level",
                        "target",
                        "label",
                        "urgency_level"
                    ]

                    feature_columns = [
                        col
                        for col in training_df.columns
                        if col not in excluded
                    ]

                    for col in feature_columns:

                        if col not in input_df.columns:

                            if (
                                training_df[col].dtype
                                == "object"
                            ):

                                mode = (
                                    training_df[col]
                                    .mode()
                                )

                                if not mode.empty:
                                    value = mode.iloc[0]
                                else:
                                    value = ""

                                input_df[col] = value

                            else:

                                input_df[col] = (
                                    training_df[col]
                                    .median()
                                )

                    input_df = input_df[
                        feature_columns
                    ]

                if preprocessor is not None:

                    transformed = (
                        preprocessor.transform(
                            input_df
                        )
                    )

                    prediction = model.predict(
                        transformed
                    )

                else:

                    prediction = model.predict(
                        input_df
                    )

                result = prediction[0]

                urgency_map = {
                    1: "Emergency",
                    2: "Very Urgent",
                    3: "Urgent",
                    4: "Less Urgent",
                    5: "Non-Urgent"
                }

                try:
                    urgency = urgency_map.get(
                        int(result),
                        str(result)
                    )
                except:
                    urgency = str(result)

                st.session_state.last_prediction = urgency

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                if urgency == "Emergency":

                    st.error(
                        "🚨 EMERGENCY\n\n"
                        "Immediate attention required."
                    )

                elif urgency == "Very Urgent":

                    st.warning(
                        "⚠️ VERY URGENT\n\n"
                        "Priority attention required."
                    )

                elif urgency == "Urgent":

                    st.warning(
                        "🟠 URGENT\n\n"
                        "Timely assessment required."
                    )

                elif urgency == "Less Urgent":

                    st.info(
                        "🟡 LESS URGENT\n\n"
                        "Assessment can follow higher priority cases."
                    )

                else:

                    st.success(
                        "🟢 NON-URGENT\n\n"
                        "Routine assessment."
                    )

                st.metric(
                    "Predicted Urgency",
                    urgency
                )

                triage_record = {
                    "patient_id":
                        "PT"
                        + datetime.now().strftime(
                            "%Y%m%d%H%M%S"
                        ),
                    "name":
                        patient_name,
                    "age":
                        age,
                    "sex":
                        sex,
                    "chief_complaint":
                        chief_complaint,
                    "urgency_level":
                        urgency,
                    "status":
                        "Pending",
                    "date":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                }

                save_patient(
                    triage_record
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

        st.caption(
            "Educational ML prototype only. "
            "Do not use this prediction for actual clinical decisions."
        )


# ============================================================
# PATIENT QUEUE
# ============================================================

elif st.session_state.page == "Patient Queue":

    st.title("📋 Patient Queue")

    st.write(
        "Monitor patients waiting for assessment."
    )

    st.divider()

    patients_df = load_patients()

    if patients_df.empty:

        st.info(
            "No patients are currently in the queue."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            status_filter = st.selectbox(
                "Status",
                [
                    "All",
                    "Pending",
                    "Completed"
                ]
            )

        with col2:

            urgency_filter = st.selectbox(
                "Urgency",
                [
                    "All",
                    "Emergency",
                    "Very Urgent",
                    "Urgent",
                    "Less Urgent",
                    "Non-Urgent"
                ]
            )

        display_df = patients_df.copy()

        if (
            status_filter != "All"
            and "status" in display_df.columns
        ):

            display_df = display_df[
                display_df["status"]
                .astype(str)
                .str.lower()
                ==
                status_filter.lower()
            ]

        if (
            urgency_filter != "All"
            and "urgency_level"
            in display_df.columns
        ):

            display_df = display_df[
                display_df[
                    "urgency_level"
                ]
                .astype(str)
                .str.lower()
                ==
                urgency_filter.lower()
            ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# REPORTS
# ============================================================

elif st.session_state.page == "Reports & Analytics":

    st.title("📊 Reports & Analytics")

    st.write(
        "Monitor patient and triage statistics."
    )

    st.divider()

    patients_df = load_patients()

    if patients_df.empty:

        st.info(
            "No patient data available yet."
        )

    else:

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Total Records",
                len(patients_df)
            )

        with c2:

            if "age" in patients_df.columns:

                st.metric(
                    "Average Age",
                    round(
                        patients_df["age"].mean(),
                        1
                    )
                )

        with c3:

            if "urgency_level" in patients_df.columns:

                emergency = len(
                    patients_df[
                        patients_df[
                            "urgency_level"
                        ]
                        .astype(str)
                        .str.contains(
                            "Emergency",
                            case=False,
                            na=False
                        )
                    ]
                )

                st.metric(
                    "Emergency Cases",
                    emergency
                )

        st.write("")

        if "urgency_level" in patients_df.columns:

            st.subheader(
                "Urgency Distribution"
            )

            urgency_counts = (
                patients_df[
                    "urgency_level"
                ]
                .value_counts()
            )

            st.bar_chart(
                urgency_counts
            )

        if "sex" in patients_df.columns:

            st.subheader(
                "Patient Distribution"
            )

            sex_counts = (
                patients_df[
                    "sex"
                ]
                .value_counts()
            )

            st.bar_chart(
                sex_counts
            )


# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    st.title("⚙️ Settings")

    st.write(
        "Application and machine learning configuration."
    )

    st.divider()

    st.subheader(
        "AI Model"
    )

    if model_bundle is not None:

        st.success(
            "🟢 Model loaded successfully"
        )

    else:

        st.error(
            "🔴 Model not found"
        )

    st.write(
        f"Model location: `{MODEL_PATH}`"
    )

    st.subheader(
        "Dataset"
    )

    if training_df.empty:

        st.warning(
            "Dataset not found."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Rows",
                f"{training_df.shape[0]:,}"
            )

        with c2:

            st.metric(
                "Columns",
                training_df.shape[1]
            )

    st.subheader(
        "About"
    )

    st.write(
        "MedTriage AI is a machine-learning based "
        "healthcare triage management prototype."
    )

    st.warning(
        "This application is designed for educational "
        "and demonstration purposes only."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 MedTriage AI  •  Healthcare Management  •  AI Triage"
)