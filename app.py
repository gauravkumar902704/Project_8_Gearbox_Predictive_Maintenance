import os
import tempfile
import streamlit as st

from src.predict_model import predict

st.set_page_config(
    page_title="Gearbox Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.sidebar.title("⚙️ Gearbox ML Project")

st.sidebar.info("""
### Models Used

- Decision Tree
- Random Forest
- Logistic Regression
- KNN
- Support Vector Machine
- Gaussian Naive Bayes
""")

st.sidebar.success("Dataset : Gearbox Vibration Dataset")

st.sidebar.markdown("---")

st.sidebar.write("Developed by")

st.sidebar.write("**Gaurav Kumar**")

st.title("⚙️ Gearbox Predictive Maintenance")

st.markdown("""
### Industrial Fault Diagnosis using Machine Learning

Upload a gearbox vibration **(.txt)** file to predict the gearbox condition.

---
""")


st.write(
    "Upload a gearbox vibration (.txt) file for health prediction."
)

uploaded_file = st.file_uploader(
    "Choose vibration file",
    type=["txt", "csv"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:

        tmp.write(uploaded_file.read())
        temp_file = tmp.name

    if st.button("Predict"):

        prediction, confidence, features = predict(temp_file)

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.success("✅ Healthy Gearbox")
            else:
                st.error("❌ Broken Tooth Detected")

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.divider()

        st.subheader("📈 Model Accuracy Comparison")

        st.image(
            "Images/accuracy_comparison.png",
            caption="Accuracy Comparison of Machine Learning Models",
            width=700
        )

        st.subheader("📊 Extracted Features")

        st.dataframe(
            features,
            use_container_width=True
        )

        os.remove(temp_file)