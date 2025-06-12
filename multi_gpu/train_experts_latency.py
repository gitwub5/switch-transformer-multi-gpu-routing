import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import seaborn as sns

# 경로 생성
for layer in ["encoder", "decoder"]:
    for gpu in ["1080Ti", "A6000"]:
        os.makedirs(f'{layer}/{gpu}', exist_ok=True)
        os.makedirs(f'results/{layer}/{gpu}', exist_ok=True)

# 데이터 로딩 및 전처리
df = pd.read_csv("../outputs/merged_latency_comparison.csv")
df["layer_expert"] = df["layer_type"] + "_" + df["layer_index"].astype(str) + "_" + df["expert_id"].astype(str)
df["token_count_log"] = np.log1p(df["token_count"])

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

df = remove_outliers_iqr(df, "latency_1080Ti")
df = remove_outliers_iqr(df, "latency_A6000")

summary = []

for expert in df["layer_expert"].unique():
    layer_type = expert.split("_")[0]
    for gpu_col, gpu_name in [("latency_1080Ti", "1080Ti"), ("latency_A6000", "A6000")]:
        sub_df = df[df["layer_expert"] == expert].copy()
        if len(sub_df) < 5:
            print(f"⚠️ {expert} ({gpu_name}) 샘플 부족으로 스킵")
            continue

        sub_df["latency_log10"] = np.log10(sub_df[gpu_col])
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(sub_df[["latency_log10"]])

        X = sub_df[["token_count_log"]].values
        y = y_scaled.ravel()
        sample_count = len(sub_df)
        print(f"🔍 {expert} ({gpu_name}) 샘플 개수: {sample_count}")

        # 🔽 모델 선택 로직
        if sample_count < 500:
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(include_bias=False)),
                ('ridge', Ridge())
            ])

            # 하이퍼파라미터 그리드 설정
            param_grid = {
                'poly__degree': [2, 3],        # 다항식 차수
                'ridge__alpha': [0.1, 1.0, 10] # Ridge 정규화 계수
            }

            # GridSearchCV 적용
            grid = GridSearchCV(
                pipeline,
                param_grid,
                scoring='r2',
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X, y)
            model = grid.best_estimator_
            model_type = f"Polynomial Ridge (GridSearch)"
        else:
            # SVR로 경향성만 부드럽게 추정
            param_grid = {
                'C': [1, 10, 100],
                'epsilon': [0.001, 0.01, 0.1],
                'gamma': ['scale', 'auto']
            }
            model = SVR(kernel='rbf')
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
            model_type = f"SVR (GridSearch)"
            model.fit(X, y)
        
        # if sample_count >= 4000:
        #     model = GradientBoostingRegressor(
        #                 n_estimators=200,        # 더 많은 트리 → 부드러움 증가
        #                 learning_rate=0.03,      # 낮은 학습률 → 덜 튐
        #                 max_depth=3,             # 낮은 깊이 → 과적합 방지
        #                 subsample=0.7,           # 부분 데이터 사용 → 더 일반화
        #                 random_state=42
        #             )
        # elif sample_count >= 1000:
        #     model = HistGradientBoostingRegressor(
        #                 max_iter=200,
        #                 max_depth=3,
        #                 learning_rate=0.03,
        #                 l2_regularization=1.0,
        #                 early_stopping=False,
        #                 random_state=42
        #             )
        # else:
        #     model = Ridge(alpha=1.0)

        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # 모델 및 스케일러 저장
        model_dir = f"{layer_type}/{gpu_name}"
        joblib.dump(model, f"{model_dir}/{expert}_model.joblib")
        joblib.dump(scaler, f"{model_dir}/{expert}_scaler.joblib")

        # 📈 시각화
        token_range = np.arange(1, 513, 10)
        token_log = np.log1p(token_range).reshape(-1, 1)
        pred_scaled = model.predict(token_log)
        pred_log10 = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        pred_latency = np.power(10, pred_log10)

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x="layer_expert", y="token_count", showfliers=False)
        plt.xticks(rotation=90)
        plt.title("Token Count Distribution per Expert")
        plt.xlabel("Layer Expert")
        plt.ylabel("Token Count")
        plt.tight_layout()
        plt.savefig("results/token_distribution_boxplot.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(token_range, pred_latency, label=f"{expert}", color="blue")
        plt.xlabel("Token Count")
        plt.ylabel("Predicted Latency (ms)")
        plt.title(f"{expert} on {gpu_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/{layer_type}/{gpu_name}/{expert}_curve.png")
        plt.close()

        summary.append({
            "layer_type": layer_type,
            "layer_expert": expert,
            "gpu": gpu_name,
            "samples": sample_count,
            "model_type": model_type,
            "mse": mse,
            "r2": r2
        })

# 📄 summary 저장
pd.DataFrame(summary).to_csv("results/training_summary_adaptive.csv", index=False)
print("✅ 전체 모델 학습 및 저장 완료")