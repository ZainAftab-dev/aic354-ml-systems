# AIC354 Assignment 2: Developing an End-to-End Machine Learning System

**Zain Aftab** (FA24-BSE-155), Section B, BS Software Engineering
COMSATS University Islamabad, Lahore Campus
Course: Machine Learning Fundamentals (AIC354), Instructor: Dr. Rao Muhammad Adeel Nawab

| Task | System | Data | Model | Test accuracy |
|---|---|---|---|---|
| 1 | Titanic Passenger Survival Prediction | Instructor's 100-row sample | SVC | 0.80 |
| 2 | Heart Disease Prediction (4 attributes: ChestPain, Slope, MajorVessels, Thal) | UCI Heart Disease, Cleveland | SVC | 0.75 |

## Contents
- `AIC354_Assignment2_Zain_Aftab_FA24-BSE-155.ipynb`: full notebook (also as `.html` and `.pdf`)
- `titanic/`, `heart_disease/`: datasets, encoded data, predictions and trained models (`svc_trained_model.pkl`)
- `streamlit-app/`: web app serving both models

**Live demo:** https://zain-aic354-ml-systems.streamlit.app

## Run
    pip install -r requirements.txt
    jupyter notebook        # Kernel -> Restart & Run All
    streamlit run streamlit-app/app.py
