import pandas as pd
import numpy as np

def generate_data(n=1200):
    np.random.seed(42)

    age = np.random.randint(20, 80, n)
    gender = np.random.choice([0, 1], n)  # 0=Female, 1=Male
    bmi = np.random.normal(27, 5, n)
    bp = np.random.normal(120, 15, n)
    cholesterol = np.random.normal(200, 30, n)
    glucose = np.random.normal(100, 20, n)
    smoking = np.random.choice([0, 1], n)
    activity = np.random.randint(1, 5, n)
    family_history = np.random.choice([0, 1], n)
    visits = np.random.randint(0, 10, n)

    # Medical Expense formula (controlled correlation)
    expenses = (
        age * 200 +
        bmi * 300 +
        bp * 100 +
        cholesterol * 50 +
        glucose * 80 +
        smoking * 5000 +
        family_history * 3000 +
        visits * 1000 +
        np.random.normal(0, 5000, n)
    )

    # Disease classification
    disease = (bmi > 30) | (bp > 140) | (glucose > 140)
    disease = disease.astype(int)

    # Risk Category
    risk_score = bmi + bp + cholesterol/10 + glucose/10 + smoking*10
    risk_category = pd.cut(risk_score,
                           bins=[0, 200, 300, 500],
                           labels=[0, 1, 2])  # Low, Medium, High

    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "BloodPressure": bp,
        "Cholesterol": cholesterol,
        "Glucose": glucose,
        "Smoking": smoking,
        "Activity": activity,
        "FamilyHistory": family_history,
        "Visits": visits,
        "Expenses": expenses,
        "Disease": disease,
        "RiskCategory": risk_category.astype(int)
    })

    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("healthcare_data.csv", index=False)