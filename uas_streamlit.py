import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from numpy import array
# Fungsi untuk membagi urutan univariate menjadi sampel

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    if len(X) == 0 or len(y) == 0:
        return None, None
    return array(X), array(y)


# Mengambil data dari Yahoo Finance
def get_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Menampilkan grafik aktual dan prediksi
def show_graph(actual, predicted):
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Actual vs Predicted')
    plt.legend()
    st.pyplot()

# Menghitung Mean Absolute Percentage Error (MAPE)
def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted)

# Aplikasi Streamlit
def main():
    st.title('Data Prediksi')
    st.write('Data yang diambil berasal dari Finance.yahoo.com')
    st.write('Data yang di uji colom volume')
    stock_symbol = st.text_input('Masukkan simbol saham (contoh: AMD):')
    start_date = st.text_input('Masukkan tanggal mulai (YYYY-MM-DD):')
    end_date = st.text_input('Masukkan tanggal berakhir (YYYY-MM-DD):')

    if stock_symbol and start_date and end_date:
        # Ambil data dari Yahoo Finance
        data = get_data(stock_symbol, start_date, end_date)
        st.write('Data Saham:')
        st.dataframe(data)

        # Preprocessing data
        data_open = data['Open']
        n_steps = 5
        X, y = split_sequence(data_open, n_steps)
        df_X = pd.DataFrame(X, columns=['t-'+str(i) for i in range(n_steps-1, -1, -1)])
        df_y = pd.DataFrame(y, columns=['t+1 (prediction)'])
        df = pd.concat([df_X, df_y], axis=1)
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(df_X)
        X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=0)

        # Model KNN
        model_knn = KNeighborsRegressor(n_neighbors=5)
        model_knn.fit(X_train, y_train)
        y_pred_knn = model_knn.predict(X_test)
        mape_knn = calculate_mape(y_test, y_pred_knn)
        st.subheader('KNN')
        st.write('MAPE:', mape_knn)
        show_graph(y_test, y_pred_knn)

        # Model Decision Tree
        model_dt = DecisionTreeRegressor()
        model_dt.fit(X_train, y_train)
        y_pred_dt = model_dt.predict(X_test)
        mape_dt = calculate_mape(y_test, y_pred_dt)
        st.subheader('Decision Tree')
        st.write('MAPE:', mape_dt)
        show_graph(y_test, y_pred_dt)
    else:
        st.warning('Silakan masukkan simbol saham dan tanggal untuk melanjutkan.')


if __name__ == '__main__':
    main()
