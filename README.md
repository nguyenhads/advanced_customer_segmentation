# Customer Segmentation Project

Dự án phân khúc khách hàng sử dụng thuật toán K-means clustering để phân tích hành vi mua sắm và chia nhóm khách hàng.

## Mục tiêu

Phân tích dữ liệu giao dịch của khách hàng để:

- Hiểu rõ hành vi mua sắm của khách hàng
- Chia khách hàng thành các nhóm có đặc điểm tương tự
- Đưa ra chiến lược marketing phù hợp với từng nhóm

## Dữ liệu

- **Nguồn**: Dữ liệu giao dịch công ty bán lẻ trực tuyến UK (2010-2011)
- **Quy mô**: 541,909 giao dịch từ 4,372 khách hàng
- **Đặc điểm**: Giao dịch quà tặng và đồ gia dụng độc đáo

## Cấu trúc dự án

```
├── data/
│   ├── raw/                    # Dữ liệu thô
│   └── processed/              # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb     # Làm sạch dữ liệu và EDA
│   ├── 02_feature_engineering.ipynb  # Xây dựng features
│   └── 03_modeling.ipynb             # Mô hình clustering
├── src/
│   └── clustering_library.py         # Thư viện chính
├── docs/
│   └── project_description.md        # Mô tả chi tiết dự án
└── requirements.txt                  # Dependencies
```

## Bắt đầu nhanh

1. **Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

2. **Chạy notebooks theo thứ tự:**
   - `01_cleaning_and_eda.ipynb` - Làm sạch và khám phá dữ liệu
   - `02_feature_engineering.ipynb` - Tạo features RFM
   - `03_modeling.ipynb` - Xây dựng mô hình clustering

## Công nghệ sử dụng

- **Python**
- **Pandas** - Xử lý dữ liệu
- **Scikit-learn** - Machine learning
- **Matplotlib/Seaborn** - Visualization
- **NumPy** - Tính toán số học

## Tài liệu

Chi tiết về phương pháp và lý thuyết được mô tả trong `docs/project_description.md`

## Tác giả

- Tác giả: Nguyễn Thái Hà
- Dự án được phát triển như một phần của khóa học AIO 2025.
