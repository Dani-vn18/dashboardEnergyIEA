import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Configuración básica de la página (única para toda la aplicación)
st.set_page_config(
    page_title="Electricidad - Proyecto Bootcamp ARIMA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def vista_prediccion():
    st.header('Estadísticas de Electricidad - Predicción de Value con ARIMA y normalización Min-Max')
    st.warning('Se debe cargar un archivo Excel con una hoja llamada "TableData" y las columnas requeridas.')

    # Cargar datos desde Excel
    uploaded_file = st.file_uploader("Elige un archivo Excel", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="TableData")

            # Convertir la columna 'Time' a datetime
            df['Time'] = pd.to_datetime(df['Time'])

            # Filtros de selección
            country = st.selectbox("Selecciona un país:", df['Country'].unique())
            balance = st.selectbox("Selecciona un Balance:", df['Balance'].unique())
            product = st.selectbox("Selecciona un Producto:", df['Product'].unique())
            value_col = st.selectbox("Selecciona la columna de valor:", ['Value'])

            # Filtrar DataFrame
            df_filtered = df[(df['Country'] == country) &
                             (df['Balance'] == balance) &
                             (df['Product'] == product)]

            # Controles para el periodo de tiempo
            min_date = df_filtered['Time'].min()
            max_date = df_filtered['Time'].max()

            start_year = st.selectbox("Año de inicio:", range(min_date.year, max_date.year + 1), index=0)
            start_month = st.selectbox("Mes de inicio:", range(1, 13), index=min_date.month - 1)
            start_date = pd.to_datetime(f'{start_year}-{start_month}-01')

            end_year = st.selectbox("Año de fin:", range(start_date.year, max_date.year + 1), 
                                    index=len(range(start_date.year, max_date.year + 1))-1)
            end_month = st.selectbox("Mes de fin:", range(1, 13), index=max_date.month - 1)
            end_date = pd.to_datetime(f'{end_year}-{end_month}-01')

            # Filtrar por periodo de tiempo
            df_filtered = df_filtered[(df_filtered['Time'] >= start_date) & (df_filtered['Time'] <= end_date)]

            # Opción para eliminar valores atípicos (IQR)
            if st.checkbox("Eliminar valores atípicos (IQR)"):
                Q1 = df_filtered[value_col].quantile(0.25)
                Q3 = df_filtered[value_col].quantile(0.75)
                IQR = Q3 - Q1
                df_filtered = df_filtered[~((df_filtered[value_col] < (Q1 - 1.5 * IQR)) | 
                                            (df_filtered[value_col] > (Q3 + 1.5 * IQR)))]

            # Mostrar DataFrame filtrado
            st.dataframe(df_filtered)

            # Parámetros para el modelo ARIMA y el pronóstico
            periodos_futuros = st.slider('Periodos a predecir (meses)', 1, 24, 1)
            p = st.number_input("ARIMA: p (orden autorregresivo)", min_value=0, max_value=10, value=1, step=1)
            d = st.number_input("ARIMA: d (orden de diferenciación)", min_value=0, max_value=2, value=1, step=1)
            q = st.number_input("ARIMA: q (orden de media móvil)", min_value=0, max_value=10, value=1, step=1)

            if st.button('Ejecutar predicción', type='primary'):
                try:
                    # Ordenar los datos por fecha y reiniciar el índice
                    df_filtered = df_filtered.sort_values('Time').reset_index(drop=True)

                    # Normalización Min-Max de la columna Value
                    scaler = MinMaxScaler()
                    df_filtered['Value_norm'] = scaler.fit_transform(df_filtered[[value_col]])

                    # Preparar la serie temporal y las fechas
                    series = df_filtered['Value_norm']
                    dates = df_filtered['Time']

                    # Ajustar el modelo ARIMA en los datos normalizados
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()

                    # Realizar predicción in-sample (desde el punto donde se pueden obtener predicciones)
                    start_pred = d  # Los primeros 'd' valores pueden no tener predicción
                    pred_in_sample = model_fit.predict(start=start_pred, end=len(series)-1)

                    # Invertir la normalización para obtener los valores en escala original
                    actual_in_sample = scaler.inverse_transform(series[start_pred:].values.reshape(-1, 1)).flatten()
                    pred_in_sample_original = scaler.inverse_transform(pred_in_sample.values.reshape(-1, 1)).flatten()

                    # Calcular métricas de error con la predicción in-sample
                    mae = mean_absolute_error(actual_in_sample, pred_in_sample_original)
                    rmse = np.sqrt(mean_squared_error(actual_in_sample, pred_in_sample_original))
                    r2 = r2_score(actual_in_sample, pred_in_sample_original)

                    # Pronóstico de periodos futuros
                    forecast_norm = model_fit.forecast(steps=periodos_futuros)
                    forecast_original = scaler.inverse_transform(forecast_norm.values.reshape(-1, 1)).flatten()

                    # Generar fechas para el pronóstico futuro (datos mensuales)
                    last_date = dates.iloc[-1]
                    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                                 periods=periodos_futuros, freq='MS')

                    # Preparar DataFrames para resultados:
                    # Datos reales
                    df_actual = pd.DataFrame({
                        'ds': dates,
                        'y': df_filtered[value_col],
                        'Tipo': 'Real'
                    })

                    # Pronóstico futuro
                    df_forecast = pd.DataFrame({
                        'ds': future_dates,
                        'y': forecast_original,
                        'Tipo': 'Predicción'
                    })

                    # Predicción in-sample (para evaluar el modelo)
                    df_insample = pd.DataFrame({
                        'ds': dates[start_pred:],
                        'y': pred_in_sample_original,
                        'Tipo': 'Predicción In-Sample'
                    })

                    # Combinar datos reales y pronóstico futuro para la visualización
                    df_resultado = pd.concat([df_actual, df_forecast])

                    # Mostrar resultados en pestañas
                    tab1, tab2 = st.tabs(['Resultado', 'Gráficos'])
                    with tab1:
                        col1, col2 = st.columns([30, 70])
                        with col1:
                            st.subheader("Tabla de resultados")
                            st.dataframe(df_resultado)
                            csv = df_resultado.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Descargar resultado como CSV",
                                data=csv,
                                file_name=f'prediccion_arima_{uploaded_file.name}',
                                mime='text/csv'
                            )

                            # Mostrar métricas de error
                            metrics_df = pd.DataFrame({
                                'Métrica': ['MAE', 'RMSE', 'R2'],
                                'Valor': [mae, rmse, r2],
                                'Descripción': [
                                    f"Error promedio de {mae:.2f} GWh.",
                                    f"Error cuadrático medio de {rmse:.2f} GWh.",
                                    f"R2 de {r2:.2f} (cercano a 1 indica buen ajuste)."
                                ]
                            })
                            st.subheader("Métricas de Precisión (In-Sample)")
                            st.dataframe(metrics_df)

                        with col2:
                            st.write("Gráfica de la serie real y el pronóstico de ARIMA:")
                            fig = px.line(df_resultado, x='ds', y='y', color='Tipo')
                            st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.subheader("Comparación: Real vs Predicción In-Sample")
                        df_compare = df_actual.merge(df_insample, on='ds', how='inner', suffixes=('_Real', '_Pred'))
                        fig2 = px.line(df_compare, x='ds', y=['y_Real', 'y_Pred'], 
                                       labels={'value': 'Valor', 'variable': 'Tipo'})
                        st.plotly_chart(fig2, use_container_width=True)

                except Exception as e:
                    st.error(f"Error durante la predicción: {e}")
        except Exception as e:
            st.error(f"Error al cargar el archivo Excel o procesar los datos: {e}")
    else:
        st.write("Por favor, sube un archivo Excel para comenzar.")

def vista_graficos():
    st.header("Vista de Gráficos")
    st.warning('Se debe cargar un archivo Excel con una hoja llamada "TableData" y las columnas requeridas.')

    uploaded_file = st.file_uploader("Elige un archivo Excel", type=["xlsx"], key="graficos")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, sheet_name="TableData")
            
            # Convertir la columna 'Time' a datetime si existe
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
            else:
                st.error("El archivo debe contener la columna 'Time'.")
                return

            # Verificar que 'Value' esté presente
            if 'Value' not in df.columns:
                st.error("El archivo debe contener la columna 'Value'.")
                return

            # Filtros múltiples (sin selección por defecto)
            countries = st.multiselect(
                "Selecciona países",
                options=df['Country'].unique(),
                default=[]
            )
            balances = st.multiselect(
                "Selecciona balances",
                options=df['Balance'].unique(),
                default=[]
            )
            products = st.multiselect(
                "Selecciona productos",
                options=df['Product'].unique(),
                default=[]
            )

            # Filtrar el DataFrame según selecciones
            df_filtered = df[
                df['Country'].isin(countries) &
                df['Balance'].isin(balances) &
                df['Product'].isin(products)
            ]

            # Seleccionar el tipo de gráfico
            grafico_tipo = st.selectbox("Selecciona el tipo de gráfico", ["Línea", "Barras"])

            if grafico_tipo == "Línea":
                # Gráfico de línea: una serie distinta por cada combinación de País, Balance y Producto
                fig = px.line(
                    df_filtered,
                    x="Time",
                    y="Value",
                    color="Country",       # Diferencia por País
                    line_dash="Balance",   # Diferencia por Balance (estilo de línea)
                    symbol="Product",      # Diferencia por Producto (símbolo)
                    title="Valor a lo largo del tiempo"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif grafico_tipo == "Barras":
                # Agrupar por País, Balance y Producto para calcular promedio
                df_grouped = df_filtered.groupby(["Country", "Balance", "Product"])['Value'].mean().reset_index()

                # Gráfico de barras con facetas por Balance y color por País
                fig = px.bar(
                    df_grouped,
                    x="Product",
                    y="Value",
                    color="Country",
                    facet_col="Balance",
                    title="Valor promedio por País, Balance y Producto"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.write("Por favor, sube un archivo Excel para visualizar los gráficos.")

def pagina_analisis():
    st.header("Vista de Análisis de Datos")
    st.write("Análisis Descriptivo y Análisis Exploratorio")
    st.write("Contenido en desarrollo...")

# Menú de navegación en la barra lateral
st.sidebar.title("Navegación - Bootcamp Data Analysis")
pagina = st.sidebar.selectbox("Menú de Páginas", 
                              ["Vista de Predicción", "Vista de Gráficos", "Vista de Análisis de Datos"])

if pagina == "Vista de Predicción":
    vista_prediccion()
elif pagina == "Vista de Gráficos":
    vista_graficos()
elif pagina == "Vista de Análisis de Datos":
    pagina_analisis()
