# Phân loại Thiên thạch Nguy hiểm gần Trái Đất (NASA NEO)

> Dự án môn **ADY201m – AI, Data Science with Python & SQL** (FPT Education) · Nhóm 2 · Lớp AI2101

Xây dựng pipeline học máy hoàn chỉnh để phân loại các thiên thạch gần Trái Đất (Near-Earth Objects) thành **An toàn** hoặc **Nguy hiểm** (*Potentially Hazardous Asteroid*), dựa trên dữ liệu thu thập trực tiếp từ **NASA NeoWs API**.

---

## 📑 Mục lục

1. [Tổng quan](#-tổng-quan)
2. [Dữ liệu](#-dữ-liệu)
3. [Quy trình thực hiện](#-quy-trình-thực-hiện)
4. [Kết quả](#-kết-quả)
5. [Hạn chế & lưu ý](#-hạn-chế--lưu-ý)
6. [Cài đặt & chạy lại](#-cài-đặt--chạy-lại)
7. [Sử dụng mô hình đã huấn luyện](#-sử-dụng-mô-hình-đã-huấn-luyện)
8. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
9. [Thành viên nhóm](#-thành-viên-nhóm)

---

## 🔭 Tổng quan

| Mục | Nội dung |
|---|---|
| **Bài toán** | Phân loại nhị phân (0 = An toàn, 1 = Nguy hiểm) |
| **Nguồn dữ liệu** | [NASA NeoWs – Neo Browse API](https://api.nasa.gov/) |
| **Quy mô** | 8.000 thiên thạch thu thập → 7.993 mẫu sau làm sạch |
| **Đặc trưng đầu vào** | 6 đặc trưng vật lý và quỹ đạo |
| **Mô hình so sánh** | Logistic Regression, Random Forest, XGBoost |
| **Chỉ số chính** | F1-Score (lớp Nguy hiểm), PR-AUC |
| **Mô hình cuối** | Random Forest + tinh chỉnh ngưỡng quyết định (threshold = 0.30) |

---

## 📊 Dữ liệu

Dữ liệu được thu thập qua endpoint `neo/rest/v1/neo/browse` (400 trang) và lưu tại `thien_thach_raw.csv`.

### Các thuộc tính

| Thuộc tính | Ý nghĩa | Sử dụng |
|---|---|---|
| Mã định danh, Tên thiên thạch | Định danh đối tượng | ❌ Loại bỏ (không mang thông tin vật lý) |
| Độ bất định quỹ đạo | `orbit_uncertainty` | ✅ |
| Cấp sao tuyệt đối | `absolute_magnitude_h` (H) | ✅ |
| Đường kính vật thể | Đường kính ước tính tối đa (km) | ✅ |
| Khoảng cách giao cắt quỹ đạo tối thiểu | MOID (AU) | ✅ |
| Độ lệch tâm | Eccentricity | ✅ |
| Bán trục lớn | Semi-major axis | ✅ |
| Độ nghiêng quỹ đạo | Inclination | ❌ Loại bỏ (tương quan ≈ 0 với nhãn) |
| Chu kỳ quỹ đạo | Orbital period | ❌ Loại bỏ (đa cộng tuyến, r = 0.95 với Bán trục lớn) |
| **Nguy hiểm** | `is_potentially_hazardous_asteroid` | 🎯 Biến mục tiêu |

### Chất lượng dữ liệu

- Không có dòng trùng lặp (theo ID và theo thuộc tính).
- 7 dòng thiếu đồng thời *Cấp sao tuyệt đối* và *Đường kính vật thể* → loại bỏ.
- Kích thước: `(8000, 11)` → `(7993, 9)`.

---

## 🧪 Quy trình thực hiện

### 1. Thu thập dữ liệu
Gọi NASA API (API key lưu trong Colab Secrets với tên `NASA_API_KEY`), lưu cache vào CSV để tránh gọi lại.

### 2. Làm sạch dữ liệu
Bỏ cột định danh và loại 7 dòng khuyết → xuất `thien_thach_clean.csv`.

### 3. Phân tích khám phá (EDA)

| Biểu đồ | Phát hiện chính | Quyết định |
|---|---|---|
| **Box plot** | Các đặc trưng lệch phải nặng, nhiều outlier (cao nhất: Đường kính 6.91%, MOID 2.08%). Đây là số đo vật lý thực, không phải nhiễu | Giữ 100% outlier; ưu tiên mô hình dạng cây |
| **Count plot** | Mất cân bằng nhãn: An toàn 84.14% (6.725) vs Nguy hiểm 15.86% (1.268), tỉ lệ ≈ 5.3 : 1 | Dùng F1-Score & PR-AUC thay Accuracy/ROC-AUC; dùng `class_weight` / `scale_pos_weight` |
| **Heatmap tương quan** | `Bán trục lớn` ↔ `Chu kỳ quỹ đạo` r = 0.95; MOID (r = −0.34) và Cấp sao (r = −0.22) tương quan âm mạnh nhất với nhãn | Loại bỏ 2 đặc trưng dư thừa/không liên quan |
| **Scatter plot** (H vs MOID) | Lớp Nguy hiểm nằm gọn trong vùng MOID ≤ 0.05 AU và H ≤ 22.57, tạo ranh giới dạng hình chữ nhật | Thuật toán dạng cây phù hợp để học ranh giới này |

### 4. Tiền xử lý
- Chọn 6 đặc trưng cuối cùng.
- Chia **80% Train / 20% Test**, `stratify=y`, `random_state=42`.
- `RobustScaler` (chống outlier) — chỉ áp dụng cho Logistic Regression; các mô hình cây dùng dữ liệu gốc.

### 5. Huấn luyện mô hình

| Mô hình | Xử lý mất cân bằng |
|---|---|
| Logistic Regression (baseline) | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` ≈ 5.3 |

### 6. Tinh chỉnh ngưỡng
Quét đường cong Precision–Recall để chọn ngưỡng xác suất tối đa hóa F1-Score.

---

## 🏆 Kết quả

### So sánh mô hình (ngưỡng mặc định 0.5, lớp Nguy hiểm, tập Test 1.599 mẫu)

| Mô hình | PR-AUC | Precision | Recall | F1-Score | Bỏ sót (FN) | Báo nhầm (FP) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9621 | 0.7014 | 0.9803 | 0.8177 | 5 | 106 |
| Random Forest | 0.9998 | 0.9882 | 0.9882 | 0.9882 | 3 | 3 |
| **XGBoost** | 0.9997 | 0.9922 | 0.9961 | **0.9941** | 1 | 2 |

### Sau khi tinh chỉnh ngưỡng

| Mô hình | Ngưỡng tối ưu | F1-Score | FN | FP |
|---|:---:|:---:|:---:|:---:|
| **Random Forest** ⭐ | 0.3000 | 0.9941 | **0** | 3 |
| XGBoost | 0.6125 | 0.9941 | 1 | 2 |

**Mô hình được chọn: Random Forest (threshold = 0.30)** — đạt F1 ngang XGBoost nhưng **không bỏ sót thiên thạch nguy hiểm nào** (FN = 0), ưu tiên phù hợp với bài toán cảnh báo rủi ro.

---

## ⚠️ Hạn chế & lưu ý

- **Ranh giới nhãn gần như tất định.** Nhãn "Nguy hiểm" của NASA được định nghĩa dựa trên MOID và cấp sao H, hai đặc trưng này lại nằm trong đầu vào. Vì vậy điểm số rất cao (PR-AUC ≈ 0.999) phản ánh việc mô hình học lại quy tắc định nghĩa, chứ không hẳn là khả năng dự báo độc lập.
- **Ngưỡng được tinh chỉnh trên tập Test.** Để đánh giá khách quan hơn, nên chọn ngưỡng trên tập validation hoặc bằng cross-validation, sau đó mới báo cáo trên tập Test.
- **Chỉ một lần chia Train/Test.** Chưa dùng cross-validation hay tối ưu siêu tham số (GridSearch/Optuna).
- Số mẫu lớp Nguy hiểm trên tập Test chỉ 254, nên chênh lệch 1–3 mẫu có thể do ngẫu nhiên.

---

## 🚀 Cài đặt & chạy lại

Notebook được viết cho **Google Colab** (dùng `google.colab.userdata`).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy requests matplotlib seaborn scikit-learn xgboost joblib
```

1. Lấy API key miễn phí tại <https://api.nasa.gov/>.
2. Mở `ADY201m_Group2_AI2101.ipynb` trên Google Colab.
3. Thêm key vào **Secrets** (biểu tượng 🔑) với tên `NASA_API_KEY`.
4. Chạy toàn bộ notebook (`Runtime → Run all`).

> Nếu đã có sẵn `thien_thach_raw.csv`, notebook sẽ đọc file này và bỏ qua bước gọi API.
> Khi chạy ngoài Colab, thay phần `google.colab` bằng biến môi trường cho API key.

---

## 🧰 Sử dụng mô hình đã huấn luyện

File `nasa_asteroid_rf_final.pkl` chứa mô hình, ngưỡng tối ưu và danh sách đặc trưng.

```python
import joblib
import pandas as pd

pkg = joblib.load("nasa_asteroid_rf_final.pkl")
model, threshold, features = pkg["model"], pkg["threshold"], pkg["features"]

# Ví dụ một thiên thạch (giá trị minh họa)
sample = pd.DataFrame([{
    "Độ bất định quỹ đạo": 0,
    "Cấp sao tuyệt đối": 15.27,
    "Đường kính vật thể": 5.25,
    "Khoảng cách giao cắt quỹ đạo tối thiểu": 0.0295,
    "Độ lệch tâm": 0.3355,
    "Bán trục lớn": 1.2458,
}])[features]

proba = model.predict_proba(sample)[:, 1][0]
print("Xác suất nguy hiểm:", round(proba, 4))
print("Kết luận:", "NGUY HIỂM" if proba >= threshold else "An toàn")
```

---

## 📁 Cấu trúc thư mục

```
.
├── ADY201m_Group2_AI2101.ipynb    # Notebook chính
├── thien_thach_raw.csv            # Dữ liệu thô từ NASA API
├── thien_thach_clean.csv          # Dữ liệu sau làm sạch
├── nasa_asteroid_rf_final.pkl     # Mô hình Random Forest + ngưỡng
└── README.md
```

---

## 👥 Thành viên nhóm

| Họ tên | MSSV | Vai trò |
|---|---|---|
| _Thành viên 1_ | _…_ | _…_ |
| _Thành viên 2_ | _…_ | _…_ |
| _Thành viên 3_ | _…_ | _…_ |

---

## 📚 Tham khảo

- [NASA Open APIs – NeoWs (Near Earth Object Web Service)](https://api.nasa.gov/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 Giấy phép

Dự án phục vụ mục đích học tập. Thêm file `LICENSE` (ví dụ MIT) nếu bạn muốn công khai giấy phép.
