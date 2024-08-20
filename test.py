import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from tqdm import tqdm

# ハイパーパラメータの設定
look_back = 30
test_size = 0.2
n_splits = 5
epochs = 100
batch_size = 32

# 株価データの読み込み
df = pd.read_csv("stock_data.csv")

# 前処理
df.dropna(inplace=True)
df = df[["Open", "High", "Low", "Close", "Volume"]]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# 特徴量とターゲットの作成
X, y = [], []
for i in range(len(scaled_data) - look_back - 1):
    a = scaled_data[i:(i + look_back), :]
    X.append(a)
    y.append(1 if df["Close"][i + look_back + 1] > df["Close"][i + look_back] else 0)
X, y = np.array(X), np.array(y)

# 時系列を考慮したデータ分割
tscv = TimeSeriesSplit(n_splits=n_splits)

# LSTMモデルの構築
def create_lstm_model(units=50, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 5), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# GRUモデルの構築
def create_gru_model(units=50, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(GRU(units, input_shape=(look_back, 5), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# ハイパーパラメータのグリッドサーチ
param_grid = {
    'units': [50, 100],
    'dropout_rate': [0.2, 0.3],
    'optimizer': ['adam', 'rmsprop']
}

# Early Stoppingの設定
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# LSTMモデルでグリッドサーチ
lstm_model = KerasClassifier(build_fn=create_lstm_model, epochs=epochs, batch_size=batch_size, verbose=0)
lstm_grid_search = GridSearchCV(estimator=lstm_model, param_grid=param_grid, cv=tscv, n_jobs=-1)
lstm_grid_search.fit(X, y, callbacks=[early_stopping])

# GRUモデルでグリッドサーチ
gru_model = KerasClassifier(build_fn=create_gru_model, epochs=epochs, batch_size=batch_size, verbose=0)
gru_grid_search = GridSearchCV(estimator=gru_model, param_grid=param_grid, cv=tscv, n_jobs=-1)
gru_grid_search.fit(X, y, callbacks=[early_stopping])

# 最適なモデルの選択
best_model = lstm_grid_search if lstm_grid_search.best_score_ > gru_grid_search.best_score_ else gru_grid_search

# テストデータでの評価
for train_index, test_index in tqdm(tscv.split(X), total=n_splits):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# 売買シグナルの生成
signal = best_model.predict(X[-look_back:])
print(f"Buy" if signal else "Sell")
