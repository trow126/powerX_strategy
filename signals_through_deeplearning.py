import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv('price_data.csv', parse_dates=True, index_col='timestamp')
prices = df['close'].values

# 特徴量とラベルの生成
window_size = 10  # 10日間のデータを入力とする

X = []
y = []

for i in range(len(prices) - window_size):
    X.append(prices[i:i+window_size])
    y.append(prices[i+window_size])

X = np.array(X)
y = np.array(y)

# データの標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# モデルの定義
model = Sequential()
model.add(Dense(50, input_dim=window_size, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# モデルのコンパイル
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# モデルの訓練
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# テストデータに対する予測
predicted_prices = model.predict(X_test)

# 結果のプロット
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
