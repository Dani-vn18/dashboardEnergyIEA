import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# Configuración básica de la página
st.set_page_config(
    page_title="Electricidad - Proyecto Bootcamp",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header('Estadísticas de Electricidad Mensual - Predicción de Series de Tiempo con Prophet')
st.warning('Se debe cargar un archivo Excel con una hoja llamada "TableData" y las columnas requeridas. (Último Informe de la IEA)')

# Cargar los datos desde Excel
uploaded_file = st.file_uploader("Elige un archivo Excel", type=["xlsx"])
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="TableData")

        # Convertir la columna 'Time' a datetime
        df['Time'] = pd.to_datetime(df['Time'], format='%B %Y')

        # Controles de selección para filtros
        country = st.selectbox("Selecciona un país:", df['Country'].unique())
        balance = st.selectbox("Selecciona un Balance:", df['Balance'].unique())
        product = st.selectbox("Selecciona un Producto:", df['Product'].unique())
        value_col = st.selectbox("Selecciona la columna de valor:", ['Value'])

        # Filtrar el DataFrame
        df_filtered = df[
            (df['Country'] == country) &
            (df['Balance'] == balance) &
            (df['Product'] == product)
        ]

        # Mostrar el DataFrame filtrado
        st.dataframe(df_filtered)

        # Controles para Prophet
        frecuencias = ['Mes', 'Año']
        frecuenciasCodigo = ['ME', 'YE']
        parFrecuencia = st.selectbox('Frecuencia de los datos', options=frecuencias)
        parPeriodosFuturos = st.slider('Periodos a predecir', 1, 24, 1)

        if st.button('Ejecutar predicción', type='primary'):
            try:
                # Preparar datos para Prophet
                df_prophet = df_filtered[['Time', value_col]].rename(columns={'Time': 'ds', value_col: 'y'})
                m = Prophet()
                m.fit(df_prophet)
                frequencia = frecuenciasCodigo[frecuencias.index(parFrecuencia)]
                future = m.make_future_dataframe(periods=parPeriodosFuturos, freq=frequencia)
                forecast = m.predict(future)
                dfPrediccion = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(parPeriodosFuturos)

                # Visualizaciones
                tab1, tab2 = st.tabs(['Resultado', 'Gráfico Prophet'])
                df_filtered['Tipo'] = 'Real'
                dfPrediccion['Tipo'] = 'Predicción'
                dfPrediccion = dfPrediccion.rename(columns={'yhat': 'y'})
                dfResultado = pd.concat([df_filtered[['Time', value_col, 'Tipo']].rename(columns={'Time': 'ds', value_col: 'y'}), dfPrediccion[['ds', 'y', 'Tipo']]])

                with tab1:
                    c1, c2 = st.columns([30, 70])
                    with c1:
                        st.dataframe(dfResultado)
                        csv = dfResultado.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar resultado como CSV",
                            data=csv,
                            file_name=f'prediccion_{uploaded_file.name}',
                            mime='text/csv',
                            type='primary'
                        )
                    with c2:
                        fig = px.line(dfResultado, x='ds', y='y', color='Tipo')
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    fig1 = m.plot(forecast)
                    st.write(fig1)

            except Exception as e:
                st.error(f"Error durante la predicción: {e}")

    except Exception as e:
        st.error(f"Error al cargar el archivo Excel o procesar los datos: {e}")

else:
    st.write("Por favor, sube un archivo Excel para comenzar.")