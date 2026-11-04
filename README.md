# Phân loại Thiên thạch Nguy hiểm gần Trái Đất (NASA NEO)

Dự án môn ADY201m (AI, Data Science with Python & SQL), FPT Education. Nhóm 2, lớp AI2101.

Dự án xây dựng một pipeline học máy hoàn chỉnh để phân loại các thiên thạch gần Trái Đất (Near-Earth Objects) thành hai lớp: An toàn và Nguy hiểm (Potentially Hazardous Asteroid). Dữ liệu được thu thập trực tiếp từ NASA NeoWs API.

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Dữ liệu](#dữ-liệu)
3. [Quy trình thực hiện](#quy-trình-thực-hiện)
4. [Kết quả](#kết-quả)
5. [Hạn chế và lưu ý](#hạn-chế-và-lưu-ý)
6. [Cài đặt và chạy lại](#cài-đặt-và-chạy-lại)
7. [Sử dụng mô hình đã huấn luyện](#sử-dụng-mô-hình-đã-huấn-luyện)
8. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
9. [Thành viên nhóm](#thành-viên-nhóm)
10. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

## Tổng quan

| Mục | Nội dung |
|---|---|
| Bài toán | Phân loại nhị phân (0: An toàn, 1: Nguy hiểm) |
| Nguồn dữ liệu | NASA NeoWs, endpoint Neo Browse |
| Quy mô | 8.000 thiên thạch thu thập, còn 7.993 mẫu sau làm sạch |
| Đặc trưng đầu vào | 6 đặc trưng vật lý và quỹ đạo |
| Mô hình so sánh | Logistic Regression, Random Forest, XGBoost |
| Chỉ số đánh giá chính | F1-Score (lớp Nguy hiểm) và PR-AUC |
| Mô hình cuối cùng | Random Forest với ngưỡng quyết định 0.30 |

## Dữ liệu

Dữ liệu được thu thập qua endpoint `neo/rest/v1/neo/browse` (400 trang) và lưu vào `thien_thach_raw.csv`.

### Các thuộc tính

| Thuộc tính | Ý nghĩa | Sử dụng |
|---|---|---|
| Mã định danh, Tên thiên thạch | Định danh đối tượng | Loại bỏ (không mang thông tin vật lý) |
| Độ bất định quỹ đạo | `orbit_uncertainty` | Giữ lại |
| Cấp sao tuyệt đối | `absolute_magnitude_h` (H) | Giữ lại |
| Đường kính vật thể | Đường kính ước tính tối đa (km) | Giữ lại |
| Khoảng cách giao cắt quỹ đạo tối thiểu | MOID (AU) | Giữ lại |
| Độ lệch tâm | Eccentricity | Giữ lại |
| Bán trục lớn | Semi-major axis | Giữ lại |
| Độ nghiêng quỹ đạo | Inclination | Loại bỏ (tương quan gần bằng 0 với nhãn) |
| Chu kỳ quỹ đạo | Orbital period | Loại bỏ (đa cộng tuyến, r = 0.95 với Bán trục lớn) |
| Nguy hiểm | `is_potentially_hazardous_asteroid` | Biến mục tiêu |

### Chất lượng dữ liệu

- Không có dòng trùng lặp, kiểm tra theo cả ID và theo toàn bộ thuộc tính.
- 7 dòng thiếu đồng thời Cấp sao tuyệt đối và Đường kính vật thể, đã loại bỏ.
- Kích thước thay đổi từ (8000, 11) xuống (7993, 9).

## Quy trình thực hiện

### 1. Thu thập dữ liệu

Gọi NASA API với khóa lưu trong Colab Secrets (tên `NASA_API_KEY`). Kết quả được lưu vào CSV để không phải gọi lại API ở các lần chạy sau.

### 2. Làm sạch dữ liệu

Bỏ hai cột định danh và 7 dòng khuyết, sau đó xuất `thien_thach_clean.csv`.

### 3. Phân tích khám phá (EDA)

| Biểu đồ | Phát hiện chính | Quyết định |
|---|---|---|
| Box plot | Các đặc trưng lệch phải nặng và có nhiều outlier (cao nhất: Đường kính 6.91%, MOID 2.08%). Đây là số đo vật lý thực, không phải nhiễu. | Giữ toàn bộ outlier, ưu tiên mô hình dạng cây. |
| Count plot | Nhãn mất cân bằng: An toàn 84.14% (6.725 mẫu), Nguy hiểm 15.86% (1.268 mẫu), tỉ lệ khoảng 5.3 : 1. | Dùng F1-Score và PR-AUC thay cho Accuracy và ROC-AUC; dùng `class_weight` hoặc `scale_pos_weight`. |
| Ma trận tương quan | Bán trục lớn và Chu kỳ quỹ đạo có r = 0.95. MOID (r = -0.34) và Cấp sao tuyệt đối (r = -0.22) tương quan âm mạnh nhất với nhãn. | Loại hai đặc trưng dư thừa hoặc không liên quan. |
| Scatter plot (H và MOID) | Lớp Nguy hiểm nằm trọn trong vùng MOID <= 0.05 AU và H <= 22.57, tạo ranh giới hình chữ nhật. | Mô hình dạng cây phù hợp để học ranh giới này. |

### 4. Tiền xử lý

- Chọn 6 đặc trưng cuối cùng.
- Chia 80% Train và 20% Test, có `stratify=y`, `random_state=42`.
- Chuẩn hóa bằng `RobustScaler` (chịu được outlier). Chỉ áp dụng cho Logistic Regression, các mô hình cây dùng dữ liệu gốc.

### 5. Huấn luyện mô hình

| Mô hình | Cách xử lý mất cân bằng nhãn |
|---|---|
| Logistic Regression (baseline) | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` xấp xỉ 5.3 |

### 6. Tinh chỉnh ngưỡng quyết định

Quét đường cong Precision-Recall để chọn ngưỡng xác suất cho F1-Score cao nhất.

## Kết quả

### So sánh mô hình (ngưỡng mặc định 0.5, lớp Nguy hiểm, tập Test gồm 1.599 mẫu)

| Mô hình | PR-AUC | Precision | Recall | F1-Score | Bỏ sót (FN) | Báo nhầm (FP) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9621 | 0.7014 | 0.9803 | 0.8177 | 5 | 106 |
| Random Forest | 0.9998 | 0.9882 | 0.9882 | 0.9882 | 3 | 3 |
| XGBoost | 0.9997 | 0.9922 | 0.9961 | 0.9941 | 1 | 2 |

### Sau khi tinh chỉnh ngưỡng

| Mô hình | Ngưỡng tối ưu | F1-Score | FN | FP |
|---|:---:|:---:|:---:|:---:|
| Random Forest | 0.3000 | 0.9941 | 0 | 3 |
| XGBoost | 0.6125 | 0.9941 | 1 | 2 |

Mô hình được chọn là Random Forest với ngưỡng 0.30. Mô hình này đạt F1-Score bằng XGBoost nhưng không bỏ sót thiên thạch nguy hiểm nào (FN = 0), phù hợp với bài toán cảnh báo rủi ro, nơi bỏ sót một đối tượng nguy hiểm tốn kém hơn báo nhầm.

## Hạn chế và lưu ý

- Nhãn "Nguy hiểm" của NASA được xác định dựa trên MOID và cấp sao H, hai đặc trưng này lại nằm trong đầu vào của mô hình. Điểm số rất cao (PR-AUC khoảng 0.999) vì vậy chủ yếu phản ánh việc mô hình học lại quy tắc định nghĩa, chưa phải khả năng dự báo độc lập.
- Ngưỡng quyết định được tinh chỉnh trực tiếp trên tập Test, nên kết quả cuối có thể hơi lạc quan. Cách đánh giá chặt chẽ hơn là chọn ngưỡng trên tập validation hoặc bằng cross-validation, rồi mới báo cáo trên tập Test.
- Dự án mới chia Train/Test một lần, chưa dùng cross-validation và chưa tối ưu siêu tham số (GridSearch, Optuna).
- Tập Test chỉ có 254 mẫu lớp Nguy hiểm, nên chênh lệch 1 đến 3 mẫu giữa các mô hình có thể do ngẫu nhiên.

## Cài đặt và chạy lại

Notebook được viết cho Google Colab (dùng `google.colab.userdata`).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy requests matplotlib seaborn scikit-learn xgboost joblib
```

1. Đăng ký API key miễn phí tại https://api.nasa.gov/.
2. Mở `ADY201m_Group2_AI2101.ipynb` trên Google Colab.
3. Thêm key vào mục Secrets với tên `NASA_API_KEY`.
4. Chạy toàn bộ notebook (Runtime, Run all).

Nếu đã có sẵn `thien_thach_raw.csv`, notebook sẽ đọc file này và bỏ qua bước gọi API. Khi chạy ngoài Colab, cần thay phần `google.colab` bằng biến môi trường để đọc API key.

## Sử dụng mô hình đã huấn luyện

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
print("Kết luận:", "Nguy hiểm" if proba >= threshold else "An toàn")
```

## Cấu trúc thư mục

```
.
├── ADY201m_Group2_AI2101.ipynb    # Notebook chính
├── thien_thach_raw.csv            # Dữ liệu thô từ NASA API
├── thien_thach_clean.csv          # Dữ liệu sau làm sạch
├── nasa_asteroid_rf_final.pkl     # Mô hình Random Forest và ngưỡng
└── README.md
```

## Tài liệu tham khảo

- NASA Open APIs, NeoWs (Near Earth Object Web Service): https://api.nasa.gov/
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- XGBoost Documentation: https://xgboost.readthedocs.io/

