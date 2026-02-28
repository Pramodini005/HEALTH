import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def train_models(df):
    X = df.drop(["Expenses", "Disease", "RiskCategory"], axis=1)

    # -------- Linear Regression --------
    y_exp = df["Expenses"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_exp, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    r2 = r2_score(y_test, lr.predict(X_test))

    # -------- Decision Tree --------
    y_dis = df["Disease"]
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_dis.loc[X_train.index])
    acc_dt = accuracy_score(y_dis.loc[X_test.index], dt.predict(X_test))

    # -------- KNN --------
    y_risk = df["RiskCategory"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y_risk)
    acc_knn = accuracy_score(y_risk, knn.predict(X_scaled))

    # -------- KMeans --------
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(X_scaled)

    return lr, dt, knn, kmeans, scaler, r2, acc_dt, acc_knn