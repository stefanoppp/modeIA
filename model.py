import pymysql
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm

def connect_to_database():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='datasettif'
    )
    return conn

def fetch_data_from_db(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT precio, aceitunas, inflacion, fecha FROM datos")
    data = cursor.fetchall()
    return data

def preprocess_data(data):
    aceitunas = []
    inflaciones = []
    fechas = []
    precios = []  # Lista separada para almacenar el precio del aceite de oliva

    for fila in data:
        precios.append(fila[0])  
        aceitunas.append(fila[1])
        inflaciones.append(fila[2])
        fechas.append(datetime_to_float(fila[3]))

    array_compuesto = []
    for i in range(len(aceitunas)):
        aux_array = [aceitunas[i], inflaciones[i], fechas[i]]
        array_compuesto.append(aux_array)
    return array_compuesto,precios

    # X = np.array([aceitunas, inflaciones, fechas]).T 
    # y = np.array(precios)
    # return X, y

    

def datetime_to_float(date):
    fecha_datetime = datetime.combine(date, datetime.min.time())
    fecha_timestamp = fecha_datetime.timestamp()
    return fecha_timestamp

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print("Error cuadrado medio:", mse)
    print("Coeficiente de determinación:", r2)
    print("Error absoluto medio:", mae)

def calculate_aic_bic(X, y, model):
    # Adding constant for AIC and BIC calculation
    X = sm.add_constant(X)
    model_with_const = model.fit(X, y)
    y_pred = model_with_const.predict(X)
    n = len(y)
    k = X.shape[1]
    resid = y - y_pred
    sse = np.sum(resid ** 2)
    
    L = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(sse / n)

    aic = n * np.log(sse / n) + 2 * k
    bic = -2 * L + k * np.log(n)
    print("AIC:", aic)
    print("BIC:", bic)

def plot_data(df):
    sns.pairplot(df[['Precio aceite', 'Aceitunas', 'Inflacion']])
    plt.show()

def main():
    conn = connect_to_database()
    data = fetch_data_from_db(conn)
    X, y = preprocess_data(data)

    # Entrenamiento y evaluación del modelo Random Forest Regressor
    modelo_rf = RandomForestRegressor(n_estimators=40, random_state=5)
    modelo_rf.fit(X, y)
    y_pred_rf = modelo_rf.predict(X)
    print("Random Forest Regressor:")
    evaluate_model(y, y_pred_rf)
    calculate_aic_bic(X, y, modelo_rf)

    # Entrenamiento y evaluación del modelo de Regresión Lineal--------------------------------------
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    y_pred_lineal = modelo_lineal.predict(X)
    print("\nLinear Regression:")
    evaluate_model(y, y_pred_lineal)
    calculate_aic_bic(X, y, modelo_lineal)

    # # Entrenamiento y evaluación del modelo Gradient Boosting Regressor
    modelo_gb = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=5)
    modelo_gb.fit(X, y)
    y_pred_gb = modelo_gb.predict(X)
    print("\nGradient Boosting Regressor:")
    evaluate_model(y, y_pred_gb)
    calculate_aic_bic(X, y, modelo_gb)

    # # Visualización de datos
    df = pd.DataFrame(data, columns=['Precio aceite', 'Aceitunas', 'Inflacion', 'Fecha'])
    df['Fecha'] = pd.to_datetime(df['Fecha'])

    plot_data(df)
    
if __name__ == "__main__":
    main()
