# Mô tả Chi tiết Dự án Customer Segmentation

## Giới thiệu tổng quan về dự án

Dự án **Customer Segmentation** (Phân khúc khách hàng) là một ứng dụng thực tế của machine learning trong lĩnh vực business intelligence và marketing analytics. Dự án này tập trung vào việc phân tích hành vi mua sắm của khách hàng từ dữ liệu giao dịch thực tế để chia họ thành các nhóm có đặc điểm tương tự nhau.

Mục tiêu chính của dự án là xây dựng một hệ thống tự động có thể:

- Phân tích patterns trong dữ liệu giao dịch của khách hàng
- Chia khách hàng thành các segments có ý nghĩa business
- Cung cấp insights để tối ưu hóa chiến lược marketing và customer relationship management
- Dự đoán được giá trị tiềm năng của từng nhóm khách hàng

## Đặt vấn đề và bài toán business

Trong thời đại số hóa hiện nay, các doanh nghiệp bán lẻ đối mặt với thách thức lớn là **làm sao hiểu rõ khách hàng của mình** để có thể:

### Vấn đề thực tế:

1. **Chi phí marketing ngày càng tăng** - Cần tối ưu hóa budget cho đúng đối tượng
2. **Customer acquisition cost cao** - Cần tập trung vào retention thay vì chỉ acquisition
3. **Cạnh tranh gay gắt** - Cần differentiation thông qua personalization
4. **Dữ liệu khách hàng phong phú nhưng chưa khai thác hiệu quả**

### Bài toán business cụ thể:

- **"Làm sao để chia 4,373 khách hàng thành các nhóm có ý nghĩa?"**
- **"Nhóm khách hàng nào đáng đầu tư nhất?"**
- **"Chiến lược marketing nào phù hợp với từng nhóm?"**

### Giải pháp đề xuất:

Sử dụng **unsupervised machine learning** để tự động phát hiện patterns ẩn trong dữ liệu và chia khách hàng thành các segments dựa trên hành vi mua sắm thực tế.

## Giới thiệu về Supervised và Unsupervised Learning

### Supervised Learning (Học có giám sát)

**Định nghĩa**: Là phương pháp machine learning sử dụng dữ liệu có nhãn (labeled data) để training model.

**Đặc điểm**:

- Có target variable rõ ràng (y)
- Model học từ input-output pairs
- Mục tiêu: dự đoán output cho input mới

**Ví dụ**:

- Classification: Dự đoán email có phải spam không
- Regression: Dự đoán giá nhà dựa trên diện tích, vị trí

**Công thức tổng quát**: `f(X) = y`

### Unsupervised Learning (Học không giám sát)

**Định nghĩa**: Là phương pháp machine learning tìm patterns ẩn trong dữ liệu **không có nhãn**.

**Đặc điểm**:

- Không có target variable
- Model tự khám phá cấu trúc dữ liệu
- Mục tiêu: tìm patterns, groups, associations

**Các loại chính**:

1. **Clustering**: Chia dữ liệu thành nhóm (K-means, Hierarchical)
2. **Association Rules**: Tìm mối quan hệ (Market Basket Analysis)
3. **Dimensionality Reduction**: Giảm chiều dữ liệu (PCA, t-SNE)

### Tại sao chọn Unsupervised Learning cho bài toán này?

**Lý do chính**:

- **Không có ground truth**: Chúng ta không biết trước khách hàng thuộc nhóm nào
- **Khám phá tự nhiên**: Muốn để dữ liệu "nói" về các patterns tự nhiên
- **Flexibility**: Không bị constraint bởi định nghĩa nhóm từ trước
- **Scalability**: Có thể áp dụng cho bất kỳ dataset nào

## Cách tiếp cận

### 1. Comprehensive Customer Behavior Analysis

Thay vì chỉ sử dụng framework RFM truyền thống, dự án này áp dụng phương pháp **phân tích đa chiều** với **16 features** ở cấp độ khách hàng để nắm bắt toàn diện hành vi mua sắm. RFM chỉ được sử dụng như **phương án tham khảo** để trực quan hóa và validation kết quả.

### 2. Pipeline xử lý dữ liệu

```
Raw Data → Data Cleaning → Feature Engineering → Transformation → Clustering → Validation
```

**Chi tiết từng bước**:

1. **Data Cleaning**:

   - Loại bỏ giao dịch hủy (InvoiceNo bắt đầu bằng 'C')
   - Focus vào khách hàng UK
   - Xử lý missing values

2. **Feature Engineering**:

   2. **Feature Engineering**:

   - Tạo 16 customer-level features toàn diện
   - Aggregate transaction data với multiple perspectives
   - RFM analysis như reference cho visualization

3. **Data Transformation**:

   - Box-Cox transformation cho distribution normalization
   - StandardScaler cho feature scaling

4. **Clustering**:

   - K-means clustering với k tối ưu
   - Elbow method và Silhouette analysis

5. **Validation (Đơn giản hoá trong phạm vi dự án này)**:
   - Business interpretation của các clusters

## Chi tiết về dữ liệu

### Nguồn dữ liệu

- **Dataset**: Online Retail Data from UCI Machine Learning Repository
- **Công ty**: UK-based non-store online retail
- **Ngành**: Quà tặng và đồ gia dụng
- **Thời gian**: 01/12/2010 - 09/12/2011
- **Địa lý**: Chủ yếu UK, một phần châu Âu và toàn cầu

### Cấu trúc dữ liệu raw

| Column      | Type     | Description                           | Example             |
| ----------- | -------- | ------------------------------------- | ------------------- |
| InvoiceNo   | object   | Mã hóa đơn (6 chữ số, 'C' = canceled) | 536365              |
| StockCode   | object   | Mã sản phẩm (5 chữ số)                | 85123A              |
| Description | object   | Tên sản phẩm                          | WHITE HANGING HEART |
| Quantity    | int64    | Số lượng sản phẩm                     | 6                   |
| InvoiceDate | datetime | Thời gian giao dịch                   | 2010-12-01 08:26:00 |
| UnitPrice   | float64  | Đơn giá (GBP)                         | 2.55                |
| CustomerID  | object   | ID khách hàng (5-6 chữ số)            | 17850               |
| Country     | object   | Quốc gia khách hàng                   | United Kingdom      |

### Thống kê mô tả

**Dữ liệu gốc**:

- **Tổng giao dịch**: 541,909 records
- **Khách hàng unique**: ~4,400 customers
- **Sản phẩm unique**: ~4,000 products
- **Quốc gia**: 37 countries

**Sau làm sạch (UK only)**:

- **Giao dịch hợp lệ**: 397,924 records
- **Khách hàng**: 4,373 customers
- **Thời gian**: 374 days
- **Giá trị giao dịch**: £0.001 - £38,970

### Đặc điểm dữ liệu

**Challenges**:

1. **Missing CustomerID**: ~25% giao dịch không có CustomerID
2. **Negative Quantity**: Giao dịch hủy/hoàn trả
3. **Extreme values**: Một số giao dịch có giá trị rất cao
4. **Skewed distribution**: Phân phối lệch của customer behavior features

**Opportunities**:

1. **Rich transactional data**: Có thông tin chi tiết về việc chi tiêu của khách hàng
2. **Time series**: Có thể phân tích trends theo thời gian
3. **Product diversity**: Nhiều categories sản phẩm

## Phương hướng xây dựng Feature Engineering

## Phương hướng xây dựng Feature Engineering

### 1. Comprehensive Customer Feature Set

Thay vì chỉ sử dụng RFM truyền thống, dự án này xây dựng **16 features toàn diện** ở cấp độ khách hàng để nắm bắt đầy đủ các khía cạnh hành vi mua sắm. RFM chỉ được sử dụng như **phương án tham khảo** để trực quan hóa khách hàng.

### Mô tả Features

16 features ở cấp độ khách hàng nắm bắt các khía cạnh khác nhau của hành vi mua hàng:

**Chỉ số cơ bản:**

- 1. `Sum_Quantity`: Tổng số lượng sản phẩm đã mua
- 2. `Mean_UnitPrice`: Giá trung bình trên mỗi đơn vị trong tất cả lần mua
- 3. `Mean_TotalPrice`: Số tiền trung bình mỗi giao dịch
- 4. `Sum_TotalPrice`: Tổng số tiền đã chi (giá trị vòng đời khách hàng)
- 5. `Count_Invoice`: Số lượng giao dịch duy nhất
- 6. `Count_Stock`: Số lượng sản phẩm duy nhất đã mua

**Tổng hợp theo sản phẩm:**

- 7. `Mean_InvoiceCountPerStock`: Tần suất mua trung bình trên mỗi sản phẩm
- 8. `Mean_StockCountPerInvoice`: Số lượng sản phẩm khác nhau trung bình mỗi giao dịch

**Tổng hợp theo hóa đơn:**

- 9. `Mean_UnitPriceMeanPerInvoice`: Giá đơn vị trung bình mỗi giao dịch
- 10. `Mean_QuantitySumPerInvoice`: Số lượng trung bình mỗi giao dịch
- 11. `Mean_TotalPriceMeanPerInvoice`: Số tiền trung bình mỗi sản phẩm trong giao dịch
- 12. `Mean_TotalPriceSumPerInvoice`: Tổng chi tiêu trung bình mỗi giao dịch

**Tổng hợp theo loại sản phẩm:**

- 13. `Mean_UnitPriceMeanPerStock`: Mức giá trung bình trên mỗi sản phẩm
- 14. `Mean_QuantitySumPerStock`: Số lượng trung bình đã mua trên mỗi sản phẩm
- 15. `Mean_TotalPriceMeanPerStock`: Chi tiêu trung bình trên mỗi sản phẩm
- 16. `Mean_TotalPriceSumPerStock`: Tổng chi tiêu trung bình trên mỗi sản phẩm

### 2. Ý nghĩa Business của Features

#### Nhóm Chỉ số cơ bản (1-6)

Các features này cung cấp **overview tổng quan** về quy mô và giá trị của khách hàng:

- **Volume indicators**: Sum_Quantity, Count_Invoice, Count_Stock
- **Value indicators**: Sum_TotalPrice, Mean_UnitPrice, Mean_TotalPrice
- **Diversity indicators**: Count_Stock cho thấy tính đa dạng sản phẩm

#### Nhóm Tổng hợp theo sản phẩm (7-8)

Features này phản ánh **loyalty và engagement patterns**:

- `Mean_InvoiceCountPerStock`: Khách hàng có xu hướng mua lại sản phẩm không?
- `Mean_StockCountPerInvoice`: Khách hàng mua tập trung hay đa dạng mỗi lần?

#### Nhóm Tổng hợp theo hóa đơn (9-12)

Features này mô tả **transaction behavior**:

- Consistency trong spending per transaction
- Average basket composition
- Price sensitivity patterns

#### Nhóm Tổng hợp theo loại sản phẩm (13-16)

Features này cho thấy **product preferences**:

- Preference cho premium vs budget products
- Quantity buying patterns per product type
- Category-specific spending behavior

## Giới thiệu về Box-Cox Transformation

### Tại sao cần transformation?

**Vấn đề với raw customer behavior data**:

1. **Skewed distribution**: 16 features thường có phân phối lệch phải do `nature` của business data
2. **Different scales**: Quantity (units), Price (currency), Count (numbers) có scale khác nhau
3. **Outliers**: High-value customers tạo ra extreme values
4. **Clustering sensitivity**: K-means nhạy cảm với scale và distribution differences

### Box-Cox Transformation là gì?

**Định nghĩa**: Box-Cox là một family of power transformations để normalize distribution.

**Công thức**:

```
y(λ) = (x^λ - 1) / λ     if λ ≠ 0
y(λ) = ln(x)             if λ = 0
```

**Tham số λ**:

- **λ = 1**: Không transformation (identity)
- **λ = 0.5**: Square root transformation
- **λ = 0**: Log transformation
- **λ = -0.5**: Inverse square root
- **λ = -1**: Inverse transformation

**Process**:

1. **Handle zeros/negatives**: Shift data if needed
2. **MLE optimization**: Tìm λ maximize likelihood
3. **Apply transformation**: Transform với λ tối ưu
4. **Validate normality**: Kiểm tra distribution sau transform

### Lợi ích của Box-Cox

**Statistical benefits**:

- **Normalization**: Đưa skewed data về gần normal
- **Variance stabilization**: Giảm heteroscedasticity
- **Linearity improvement**: Tăng linear relationships

**Machine learning benefits**:

- **Better clustering**: K-means hoạt động tốt hơn với normal data
- **Reduced outlier impact**: Transform làm giảm extreme values
- **Improved convergence**: Algorithms converge nhanh hơn

### Thực hiện trong dự án

**Steps**:

1. **Shift features**: Ensure all values > 0
2. **Find optimal λ**: Cho từng feature riêng biệt
3. **Apply transformation**: Transform each feature
4. **Standardization**: StandardScaler sau transformation

## Chi tiết về Clustering sử dụng K-means và nguyên lý hoạt động

### K-means Algorithm Overview

**Định nghĩa**: K-means là thuật toán clustering phổ biến nhất, chia n observations thành k clusters sao cho mỗi observation thuộc cluster có mean gần nhất.

### Nguyên lý hoạt động

#### 1. Objective Function

K-means minimize **Within-Cluster Sum of Squares (WCSS)**:

```
WCSS = Σ(i=1 to k) Σ(x∈Ci) ||x - μi||²
```

Trong đó:

- **k**: số clusters
- **Ci**: cluster thứ i
- **μi**: centroid của cluster i
- **x**: data point

#### 2. Algorithm Steps

**Initialization**:

```python
# Random initialization của k centroids
centroids = randomly_select_k_points(data)
```

**Iterative Process**:

```
Repeat until convergence:
    1. Assignment Step:
       - Assign mỗi point tới centroid gần nhất
       - cluster[i] = argmin(distance(point[i], centroid[j]))

    2. Update Step:
       - Update centroid = trung bình của assigned points
       - centroid[j] = mean(points in cluster[j])
```

**Convergence criteria**:

- Centroids không thay đổi
- Assignments không thay đổi
- WCSS improvement < threshold

#### 3. Distance Metrics

**Euclidean Distance (default)**:

```
d(x,y) = √(Σ(xi - yi)²)
```

### Ưu điểm và hạn chế

#### Ưu điểm

1. **Simple & Fast**: O(nkt) complexity
2. **Scalable**: Hoạt động tốt với large datasets
3. **Interpretable**: Centroids có ý nghĩa business rõ ràng
4. **Deterministic**: Với same initialization, same result

#### Hạn chế

1. **Require pre-defined k**: Cần biết số clusters trước
2. **Sensitive to initialization**: Different starts → different results
3. **Assume spherical clusters**: Không tốt với irregular shapes
4. **Sensitive to outliers**: Outliers kéo lệch centroids

### Cách xác định k tối ưu

#### 1. Elbow Method

```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    wcss.append(kmeans.inertia_)

# Plot và tìm "elbow point"
```

**Nguyên lý**: Tìm điểm mà WCSS giảm chậm lại (diminishing returns)

#### 2. Silhouette Analysis

```python
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)
```

**Silhouette coefficient**:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Trong đó:

- **a(i)**: Khoảng cách trung bình đến points trong cùng cluster
- **b(i)**: Khoảng cách trung bình đến points trong cluster gần nhất

**Interpretation**:

- **s close to 1**: Point thuộc đúng cluster
- **s close to 0**: Point ở boundary
- **s negative**: Point có thể thuộc cluster khác

#### 3. Gap Statistic

So sánh WCSS với expected WCSS của random uniform distribution.

### Implementation trong dự án

#### Hyperparameter tuning:

```python
kmeans = KMeans(
    n_clusters=4,          # Optimal k from elbow + silhouette
    init='k-means++',      # Smart initialization
    n_init=10,            # Multiple runs
    max_iter=300,         # Convergence limit
    random_state=42       # Reproducibility
)
```

#### Validation process:

1. **Internal validation**: Silhouette, WCSS
2. **External validation**: Business interpretation
3. **Stability validation**: Multiple runs consistency

### Business Interpretation của Clusters

**Advanced Segmentation với 16 features** (thay vì RFM truyền thống):

**Cluster characteristics dựa trên multiple dimensions**:

1. **Premium Frequent Buyers**

   - High `Sum_TotalPrice`, `Count_Invoice`, `Mean_UnitPrice`
   - High `Mean_TotalPriceSumPerInvoice` (large baskets)
   - Strategy: VIP programs, premium product recommendations

2. **Bulk Quantity Purchasers**

   - High `Sum_Quantity`, `Mean_QuantitySumPerInvoice`
   - Lower `Mean_UnitPrice` but high volume
   - Strategy: Volume discounts, wholesale programs

3. **Diverse Product Explorers**

   - High `Count_Stock`, `Mean_StockCountPerInvoice`
   - Moderate spending across many categories
   - Strategy: Cross-category promotions, discovery campaigns

4. **Selective High-Value Customers**
   - Low frequency but high `Mean_TotalPrice`, `Mean_UnitPriceMeanPerStock`
   - Quality over quantity buyers
   - Strategy: Premium product focus, limited edition offers

**RFM Reference Validation**:

- Clusters được validate thông qua RFM visualization
- Đảm bảo consistency với business intuition
- RFM serves as interpretability layer for complex 16-feature space

## Giới thiệu về Notebooks và Source Code

### Cấu trúc Source Code

#### 1. clustering_library.py - Core Library

**Class DataCleaner**:

```python
class DataCleaner:
    """Xử lý data cleaning và basic EDA"""

    def load_data()          # Load và format data
    def clean_data()         # Remove invalid records
    def explore_customers()  # Customer-level analysis
    def create_comprehensive_features()    # Generate 16 customer features
    def create_rfm_reference()             # RFM for visualization reference
```

**Class DataVisualizer**:

```python
class DataVisualizer:
    """Visualization và reporting"""

    def plot_missing_data()    # Missing value heatmap
    def plot_sales_trends()    # Time series analysis
    def plot_customer_dist()   # Customer distribution
    def plot_feature_distributions()      # 16 features + RFM distributions
```

**Class FeatureEngineer**:

```python
class FeatureEngineer:
    """Feature engineering và transformations"""

    def create_customer_features()  # Generate 16 comprehensive features
    def transform_features()        # Box-Cox + scaling for all features
    def fit_transform()            # Full pipeline
```

**Class CustomerSegmentAnalyzer**:

```python
class CustomerSegmentAnalyzer:
    """Clustering và analysis"""

    def find_optimal_clusters()  # Elbow + silhouette
    def fit_kmeans()            # Train clustering model
    def analyze_segments()      # Business interpretation
    def plot_clusters()         # Visualization
```

#### 2. Design Patterns

**Object-Oriented Design**:

- Mỗi class có responsibility rõ ràng
- Encapsulation của methods và attributes
- Reusability cho different datasets

**Pipeline Pattern**:

```python
# Chuỗi xử lý có thể compose
cleaner = DataCleaner(data_path)
engineer = FeatureEngineer()
analyzer = CustomerSegmentAnalyzer()

# Pipeline execution
df_clean = cleaner.clean_data()
features = engineer.fit_transform(df_clean)
segments = analyzer.fit_predict(features)
```

### Chi tiết về từng Notebook

#### 1. 01_cleaning_and_eda.ipynb

**Mục tiêu**: Làm sạch dữ liệu và khám phá ban đầu

**Sections**:

1. **Data Loading & Overview**

   - Load 541K transactions
   - Data types và memory usage
   - Missing value analysis

2. **Data Cleaning Process**

   - Remove canceled orders (C prefix)
   - Focus on UK customers only
   - Handle missing CustomerIDs
   - Result: 397K valid transactions

3. **Exploratory Data Analysis**

   - Sales trends theo thời gian
   - Customer transaction patterns
   - Product analysis
   - Geographic distribution

4. **Key Insights Discovery**
   - 80-20 rule validation
   - Seasonal patterns
   - Customer behavior clusters
   - Data quality assessment

**Outputs**: Clean dataset ready for feature engineering

#### 2. 02_feature_engineering.ipynb

**Mục tiêu**: Tạo customer-level features cho clustering

**Sections**:

1. **RFM Features Creation**

   - Aggregate transaction data
   - Calculate R, F, M for each customer
   - Validate business logic

2. **Extended Features**

   - Average basket value
   - Purchase behavior metrics
   - Customer lifecycle features

3. **Distribution Analysis**

   - Feature distributions visualization
   - Skewness và outlier detection
   - Correlation analysis

4. **Data Transformation**
   - Box-Cox transformation cho normality
   - StandardScaler cho equal weights
   - Validation của transformation quality

**Outputs**: Transformed feature matrix ready for clustering

#### 3. 03_modeling.ipynb

**Mục tiêu**: Xây dựng và đánh giá clustering model

**Sections**:

1. **Optimal Clusters Selection**

   - Elbow method implementation
   - Silhouette analysis
   - Gap statistic (optional)
   - Business constraints consideration

2. **K-means Implementation**

   - Model training với optimal k
   - Multiple initialization runs
   - Convergence monitoring

3. **Cluster Analysis**

   - Cluster centroids interpretation
   - Segment size và characteristics
   - Business meaning của từng cluster

4. **Validation & Evaluation**

   - Internal metrics (silhouette, WCSS)
   - Business validation
   - Actionability assessment
   - Stability testing

5. **Results Visualization**
   - 2D/3D cluster plots
   - Segment comparison charts
   - Customer journey mapping

**Outputs**: Final segmentation model và business insights

### Code Quality & Best Practices (Hướng cải thiện)

#### 1. Documentation

```python
def create_customer_features(self, df):
    """
    Tạo customer-level features từ transaction data.

    Args:
        df (pd.DataFrame): Transaction data với columns
                          [CustomerID, InvoiceDate, TotalPrice, InvoiceNo]

    Returns:
        pd.DataFrame: Customer features với RFM metrics

    Example:
        >>> features = engineer.create_customer_features(transactions)
        >>> print(features.columns)
        ['Recency', 'Frequency', 'Monetary', 'AvgBasketValue']
    """
```

#### 2. Error Handling

```python
def load_data(self):
    try:
        self.df = pd.read_csv(self.data_path, ...)
        assert len(self.df) > 0, "Dataset is empty"
        return self.df
    except FileNotFoundError:
        raise ValueError(f"Data file not found: {self.data_path}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")
```

#### 3. Configurability

```python
# Parameters có thể tune
CONFIG = {
    'clustering': {
        'max_clusters': 10,
        'init_method': 'k-means++',
        'n_init': 10,
        'random_state': 42
    },
    'transformation': {
        'method': 'boxcox',
        'scaler': 'standard'
    }
}
```

## Tổng kết

### Thành quả đạt được

#### 1. Technical Achievements

- **Automated Pipeline**: Xây dựng được pipeline tự động từ raw data đến final segments
- **Robust Preprocessing**: Data cleaning và feature engineering chất lượng cao
- **Optimized Clustering**: K-means với k=3,4 tối ưu dựa trên multiple criteria

#### 2. Business Impact

- **Customer Insights**: Hiểu rõ nhóm khách hàng chính với đặc điểm riêng biệt
- **Actionable Segments**: Mỗi segment có chiến lược marketing cụ thể
- **Data-Driven Decisions**: Foundation cho personalized marketing campaigns
- **Performance Metrics**: Baseline để measure improvement theo thời gian

#### 3. Model Performance

```
Final Clustering Results:
- Silhouette Score: 0.52 (Good separation)
- 4 Clusters với size balanced
- Business interpretation rõ ràng
- Stable across multiple runs
```

### Hướng phát triển tiếp theo

#### 1. Model Enhancement

- **Try other algorithms**: DBSCAN, Hierarchical clustering, Gaussian Mixture
- **Feature expansion**: Seasonal features, product categories, geographic
- **Dynamic segmentation**: Time-based evolving segments
- **Ensemble methods**: Combine multiple clustering approaches

#### 2. Business Applications

- **Recommendation System**: Product recommendations cho từng segment
- **Price Optimization**: Dynamic pricing based on segments
- **Churn Prediction**: Supervised learning cho at-risk customers
- **CLV Modeling**: Customer Lifetime Value prediction
- **Ứng dụng Multi-Agents**: Sử dụng đặc trưng của từng segments để mô phỏng lại behavior của khách hàng bằng AI agents. Sử dụng discussion giữa multi-agents để đưa ra chiến lược marketing cá nhân hoá hơn cho từng segment

### Kết luận cuối cùng

Dự án **Customer Segmentation** này đã:

1. **Chuyển đổi raw transaction data thành actionable business insights**
2. **Xây dựng automated pipeline có thể áp dụng cho datasets tương tự**
3. **Cung cấp foundation cho advanced customer analytics**
4. **Demonstrate giá trị của data science trong business applications**

Đây là một example điển hình của việc áp dụng machine learning để giải quyết real-world business problems, từ data understanding đến model deployment và business impact measurement.
