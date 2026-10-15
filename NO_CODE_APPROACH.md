# No-Code Approach: ML Star Constellation NASA

This document outlines the steps to build a star/constellation classification model using AutoML and No-Code GUI platforms (e.g., H2O Driverless AI, Google Vertex AI, DataRobot, or Pycaret's low-code interface).

## 🛠️ Step 1: Project Scoping (No-Code)
- **Objective:** Define the target variable (e.g., `class` for STAR/GALAXY/QUASAR or a `Constellation` column).
- **Success Metric:** Select **Macro F1-Score** or **Accuracy** as the optimization metric in the AutoML tool.
- **Task Type:** Choose "Multiclass Classification".

## 🧹 Step 2: Data Upload & Cleaning
1. Upload the `Star_classification.csv` file to the AutoML platform.
2. The platform will automatically profile the data, detecting missing values and data types.
3. **Handling Missing Values:** Use the platform's built-in "Impute Missing Values" feature (e.g., Median for numerical, Mode for categorical).
4. **Feature Selection:** Use the platform's automated feature importance chart to drop ID columns like `obj_ID`, `spec_obj_ID`, `run_ID`, etc., which have no predictive power.

## 📊 Step 3: Exploratory Data Analysis (EDA)
- Utilize the AutoML platform's automated EDA dashboard.
- Review the **Target Distribution** to check for class imbalance (e.g., too many stars vs. quasars).
- View **Correlation Matrices** to see how redshift or photometric bands correlate with the target class.

## 🤖 Step 4: AI Modelling (AutoML)
1. Select the target column (`class`).
2. Enable "Automatic Feature Engineering".
3. Choose the models to train (the platform will usually test XGBoost, Random Forest, GLM, and Deep Learning automatically).
4. Set the training time limit (e.g., 1 hour or 100 models).
5. Run the AutoML experiment.

## 📈 Step 5: Evaluation & Deployment
1. Review the **Leaderboard** to find the best-performing model based on F1-Score or LogLoss.
2. Inspect the **Confusion Matrix** provided by the GUI to see if the model confuses Quasars with Stars.
3. Export the model as a MOJO/ONNX or deploy it via the platform's one-click REST API endpoint for real-time astronomical predictions.
