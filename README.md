## Breast Cancer Prediction with Streamlit Integration

This professional-grade project tackles breast cancer prediction using a meticulously trained Multi-Layer Perceptron (MLP) classifier and leverages the scikit-learn library. Additionally, a user-friendly Streamlit application empowers real-time predictions, fostering a comprehensive solution for both model development and user interaction.

**Key Functionalities:**

* **Optimized Model Training:** Trains an MLP classifier on the renowned Wisconsin Diagnostic Breast Cancer (WDBC) dataset, meticulously optimizing hyperparameters through Grid Search to achieve the highest possible accuracy in binary classification (malignant vs. benign).
* **Rigorous Evaluation:** Performs in-depth model evaluation using classification reports, confusion matrix visualizations, and ROC curves, providing a holistic understanding of model performance and generalizability.
* **Persistent Model Storage:** Saves the trained model and scaler using joblib, ensuring their availability for future deployment and predictions.
* **Interactive Streamlit Application:** Delivers a user-centric interface where users can effortlessly input feature values and receive real-time cancer prediction results, fostering a seamless user experience.

**Project Structure:**

* **Modular Design:** The project is meticulously structured with well-defined modules for clarity and maintainability.
    * `BreastCancerModelTrainer.py`: A dedicated class encapsulates functionalities for data handling, preprocessing, feature selection, model training, evaluation, and saving.
    * `main.py`: The central script orchestrates the entire model training workflow.
    * `breast_cancer_prediction.py`: The Streamlit application script facilitates user interaction with the trained model.
    * `breast_cancer_model.pkl`: The saved trained model, ready for deployment.
    * `breast_cancer_scaler.pkl`: The saved scaler used for data preprocessing, ensuring consistency.

**Deployment Steps:**

**Model Training:**

1. **Environment Setup:**
   - **Python:** Ensure you have Python 3.x installed on your system.
   - **Libraries:** Install required libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `streamlit`) using `pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit`.

2. **Code Acquisition:**
   ```bash
   git clone https://github.com/meenu16411/breast-cancer-prediction.git
   ```

3. **Model Training Execution:**
   ```bash
   cd breast-cancer-prediction
   python main.py
   ```

This script executes the model training pipeline, encompassing data loading, preprocessing, model training, evaluation, and finally saving the trained model and scaler.

**Streamlit App Deployment:**

1. **Navigate to App Directory:**
   ```bash
   cd breast-cancer-prediction
   ```
2. **Run Streamlit App:**
   ```bash
   streamlit run breast_cancer_prediction.py
   ```

This will launch the Streamlit app in your web browser, typically at http://localhost:8501.

**Utilizing the Streamlit App:**

1. **Feature Value Input:**
   - The app presents input fields corresponding to each breast cancer feature.
   ![Screenshot 2024-12-01 204757](https://github.com/user-attachments/assets/d9837d35-8444-4f55-b158-1c5805f3f521)


     

2. **Prediction Generation:**
   - Click the "Predict" button.

3. **Result Visualization:**
   - The app displays a success message indicating the predicted cancer type (malignant or benign) based on the model's analysis.
     ![Screenshot 2024-12-01 204743](https://github.com/user-attachments/assets/055d432a-f441-4f51-a693-a476284c4bdd)



## Deployed APP
https://breastcancerprediction-azyhgsu5sfwyypzps2kntm.streamlit.app/


* **Disclaimer:** This model is strictly intended for educational purposes and should not be used for actual medical diagnoses. Always seek the expertise of a qualified healthcare professional for any medical concerns.










