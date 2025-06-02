import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

from AEmodel import EnsembleDenoisingAE
from sorting_function import uncertainty_sorting
from t_update_function import update_threshold

# Constants
RANDOM_SEED = 42
TRAINING_SAMPLE = 1000
VALIDATE_SIZE = 0.2
active = True

st.set_page_config(page_title="Uranie Defense System", layout="wide")
st.title("🔒 URANIE DEFENSE SYSTEM")

# -------------------------- SESSION STATE INIT --------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.EDAE = None
    st.session_state.threshold = None
    st.session_state.r_losses = np.array([])
    st.session_state.true_labels = np.array([])
    st.session_state.sorted_list = []
    st.session_state.pending_labeling = []
    st.session_state.current_label_index = 0
    st.session_state.simulation_running = False
    st.session_state.tp = st.session_state.fp = 0
    st.session_state.tn = st.session_state.fn = 0
    st.session_state.evaluated_indices = set()
    st.session_state.logs = ""
    st.session_state.y_iter = None
    st.session_state.total_items = 0

# -------------------------- LOGGING FUNCTION --------------------------
def log(msg):
    st.session_state.logs += f"{msg}\n"

# -------------------------- SETUP FUNCTION --------------------------
def setup_model():
    try:
        log("📦 Loading and training model...")
        df = pd.read_csv('workshop_s1.txt', delimiter='\t')
        df = df.drop(columns="Altered")

        X_train = df.iloc[:TRAINING_SAMPLE, :]
        X_test = df.iloc[TRAINING_SAMPLE:1050, :]
        X_train, _ = train_test_split(X_train, test_size=VALIDATE_SIZE, random_state=RANDOM_SEED)

        EDAE = EnsembleDenoisingAE()
        EDAE.build(X_train)

        unlabelled_test = pd.read_csv('workshop_s1.txt', delimiter='\t')
        x_unlabelled = unlabelled_test.iloc[700:, -1].astype(bool)  # ✅ Fix: Ensure labels are boolean
        y_unlabelled = unlabelled_test.iloc[700:, :-1]

        st.session_state.EDAE = EDAE
        st.session_state.threshold = EDAE.threshold
        st.session_state.x_unlabelled_test = x_unlabelled
        st.session_state.y_unlabelled_test = y_unlabelled
        st.session_state.y_iter = iter(y_unlabelled.iterrows())
        st.session_state.total_items = len(y_unlabelled)
        st.session_state.initialized = True
        log("✅ Model ready.")
    except Exception as e:
        log(f"❌ Error: {e}")

# -------------------------- PROCESS NEXT --------------------------
def process_next():
    try:
        index, row = next(st.session_state.y_iter)
    except StopIteration:
        st.session_state.simulation_running = False
        total = len(st.session_state.evaluated_indices)
        accuracy = (st.session_state.tp + st.session_state.tn) / total if total else 0
        log(f"\n✅ Simulation completed.\nEvaluated: {total}, Accuracy: {accuracy:.2%}")
        return

    row_df = pd.DataFrame(row).transpose()
    row_transformed = st.session_state.EDAE.Pipe.transform(row_df)
    reconstruction = st.session_state.EDAE.predict(row_transformed)
    mse = np.mean(np.power(row_transformed - reconstruction, 2), axis=1)
    mse_val = mse[0]

    true_label = st.session_state.x_unlabelled_test.loc[index]
    flagged = mse_val > st.session_state.threshold

    # ✅ Metric update fix
    if flagged and true_label:
        st.session_state.tp += 1
    elif flagged and not true_label:
        st.session_state.fp += 1
    elif not flagged and not true_label:
        st.session_state.tn += 1
    elif not flagged and true_label:
        st.session_state.fn += 1

    if flagged:
        st.session_state.pending_labeling.append({
            'mse': mse,
            'row': row_df,
            'index': index,
            'true_label': true_label,
            'model_flagged': True
        })
    else:
        st.session_state.evaluated_indices.add(index)
        log(f"✔️ No alert at index {index}, MSE={mse_val:.6f}")

    st.session_state.r_losses = np.append(st.session_state.r_losses, mse)
    st.session_state.true_labels = np.append(st.session_state.true_labels, true_label)
    st.session_state.sorted_list.append([row_df, index, mse])

    if active and len(st.session_state.sorted_list) > 100:
        r_loss = [s[2] for s in st.session_state.sorted_list]
        sorted_reconstruction = uncertainty_sorting(r_loss, st.session_state.threshold)
        top_10 = sorted_reconstruction[0][:10]
        used_indices = set()
        for x_loss in top_10:
            for sublist in st.session_state.sorted_list:
                if (sublist[1] not in used_indices and
                    isinstance(sublist[2], np.ndarray) and
                    np.array_equal(sublist[2], x_loss)):
                    st.session_state.pending_labeling.append({
                        'mse': sublist[2],
                        'row': sublist[0],
                        'index': sublist[1],
                        'true_label': st.session_state.x_unlabelled_test.loc[sublist[1]],
                        'model_flagged': False
                    })
                    used_indices.add(sublist[1])
                    break
        st.session_state.sorted_list = []

# -------------------------- LABELING --------------------------
def append_user_label(label):
    entry = st.session_state.pending_labeling[st.session_state.current_label_index]
    idx = entry['index']
    true_val = entry['true_label']

    if idx not in st.session_state.evaluated_indices:
        if label == 1:
            if true_val:
                st.session_state.tp += 1
                
            else:
                st.session_state.fp += 1

        else:
            if not true_val:
                st.session_state.tn += 1
            else:
                st.session_state.fn += 1

        st.session_state.r_losses = np.append(st.session_state.r_losses, entry['mse'])
        st.session_state.true_labels = np.append(st.session_state.true_labels, label)
        st.session_state.evaluated_indices.add(idx)
        log(f"🧑‍💼 Labeled index {idx} as {'Anomaly' if label else 'Normal'}")

    st.session_state.current_label_index += 1
    if st.session_state.current_label_index >= len(st.session_state.pending_labeling):
        update_threshold_from_labels()
    st.rerun()

def skip_label():
    st.session_state.current_label_index += 1
    if st.session_state.current_label_index >= len(st.session_state.pending_labeling):
        update_threshold_from_labels()
    st.rerun()

def update_threshold_from_labels():
    current_t = st.session_state.threshold
    st.session_state.threshold = update_threshold(
        r_losses=st.session_state.r_losses,
        true_labels=st.session_state.true_labels.astype(int),
        current_threshold=current_t
    )
    log(f"📊 Threshold updated: {current_t:.6f} → {st.session_state.threshold:.6f}")
    st.session_state.pending_labeling = []
    st.session_state.current_label_index = 0

# -------------------------- UI CONTROLS --------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Initialize Model"):
        setup_model()

with col2:
    if st.session_state.initialized and not st.session_state.simulation_running:
        if st.button("▶️ Run Simulation"):
            st.session_state.simulation_running = True

# -------------------------- LABELING UI --------------------------
if st.session_state.simulation_running and st.session_state.initialized:
    if st.session_state.pending_labeling and st.session_state.current_label_index < len(st.session_state.pending_labeling):
        entry = st.session_state.pending_labeling[st.session_state.current_label_index]
        st.markdown("### 🔍 Labeling Required")
        st.markdown(f"**MSE:** `{entry['mse'][0]:.6f}`")
        st.markdown(f"**True Label:** `{entry['true_label']}`")
        st.text(entry['row'].to_string(index=False))

        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button("✅ Anomaly"):
                append_user_label(1)
        with col4:
            if st.button("❌ Normal"):
                append_user_label(0)
        with col5:
            if st.button("⏭️ Skip"):
                skip_label()
    else:
        process_next()
        time.sleep(0.05)
        st.rerun()

# -------------------------- STATUS & LOGS --------------------------
st.markdown("### 📊 Status")
st.write(f"Threshold: {st.session_state.threshold}")
st.write(f"Evaluated: {len(st.session_state.evaluated_indices)}")

if len(st.session_state.evaluated_indices) > 0:
    accuracy = (st.session_state.tp + st.session_state.tn) / len(st.session_state.evaluated_indices)
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"TP: {st.session_state.tp}, TN: {st.session_state.tn}, FP: {st.session_state.fp}, FN: {st.session_state.fn}")

st.markdown("### 📝 Log")
st.text_area("Logs", st.session_state.logs, height=300)
