import pymysql
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():

    def conn():
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='root',
            database='datasettif'
        )
        cursor = conn.cursor()

        cursor.execute("SELECT precio, aceitunas, inflacion, fecha FROM datos")
        return cursor

    def datetime_to_float(date):
        fecha_datetime = datetime.combine(date, datetime.min.time())
        fecha_timestamp = fecha_datetime.timestamp()
        return fecha_timestamp

    def grafica():
        datos = conn()
        df = pd.DataFrame(datos, columns=['Precio aceite', 'Aceitunas', 'Inflacion', 'Fecha'])
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        sns.pairplot(df[['Precio aceite', 'Aceitunas', 'Inflacion']])
        plt.show()

    def metricas(y, y_pred):
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        print("Error cuad. medio: ", mse)
        print("Coeficiente de determinación: ", r2)
        print("Error absoluto medio: ", mae)

    precios = []
    aceitunas = []
    inflaciones = []
    fechas = []

    for fila in conn().fetchall():
        precios.append(fila[0])
        aceitunas.append(fila[1])
        inflaciones.append(fila[2])
        fechas.append(datetime_to_float(fila[3]))

    X = np.array([aceitunas, inflaciones, fechas]).T
    y = np.array(precios)

    # Entrenamiento del modelo Random forest regressor

    modelo = RandomForestRegressor(n_estimators=40, random_state=5)
    modelo.fit(X, y)

    y_pred = modelo.predict(X)

    print("Metricas para modelo Random Forest Regressor")
    metricas(y,y_pred)

    # Entrenamiento del modelo de Regresion lineal

    print("\n")
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)

    y_pred_lineal = modelo_lineal.predict(X)
    print("Metricas para modelo Linear Regression")
    metricas(y,y_pred_lineal)

    # Entrenamiento del modelo Gradient Boosting Regressor
    print("\nEntrenamiento del modelo Gradient Boosting Regressor")
    modelo_gb = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=5)
    modelo_gb.fit(X, y)
    y_pred_gb = modelo_gb.predict(X)

    print("Metricas para modelo Gradient Boosting Regressor")
    metricas(y, y_pred_gb)
    grafica()

if __name__ == "__main__":
    main()
