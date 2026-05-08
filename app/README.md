# **Customer Churn Prediction System**

## **Quick Start - How to Run, step by step**

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python -m streamlit run churn_with_batch.py
```

### 3. Open in Browser

The application will automatically open at `http://localhost:8501`

**That's it! You're ready to make predictions.**

### **OR** (streamlit cloud)

Just enter on the link below to see a streamlit cloud deploy of the model.
https://app-intelligent-techniques-churn.streamlit.app/

---

## **Features**

- **Dual Model Support**: Separate models for Internet Service and Telco customers
- **Single Predictions**: Interactive form-based predictions with real-time validation
- **Batch Predictions**: Upload CSV files to predict churn for multiple customers at once (BONUS)
- **Input Validation**: Real-time validation with error and warning messages
- **Interactive UI**: User-friendly interface with clear feedback and progress indicators
- **Risk Assessment**: Probability-based churn risk categorization (LOW/MEDIUM/HIGH/CRITICAL)
- **Actionable Recommendations**: Context-aware retention strategies based on customer data
- **Production Ready**: Easy deployment with multiple options

---

## **System Requirements**

### Minimum Requirements

- **Python**: 3.11 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 500MB free space
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

### Dependencies (already in the file `requirements.txt` )

```
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
xgboost==2.0.3
```

---

## Installation

### Local Installation

1. **Clone or download the project**

```bash
cd app
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify project structure**

```
project/
├── churn.py        # Main application
├── requirements.txt           # Python dependencies
├── models/
│   ├── xgboost_model_dataset1.pkl
│   ├── scaler_dataset1.pkl
│   ├── xgboost_model_dataset2.pkl
│   └── scaler_dataset2.pkl
└── csv_examples/                  # Sample test files for the batch section
    ├── test_dataset1_valid.csv
    ├── test_dataset2_valid.csv
    └── ...
```

---

## Using the Application

### Single Prediction Mode

1. **Select Model**: Choose Dataset 1 (Internet Service) or Dataset 2 (Telco Customer) from sidebar
2. **Navigate to Tab**: Click "Single Prediction" tab
3. **Enter Data**: Fill in customer information in the form
4. **Validate**: Review any validation errors or warnings
5. **Predict**: Click "Predict Churn Risk" button
6. **View Results**: See prediction, probability, risk level, and recommendations
7. **Clear**: Click "Clear Results" to start over

### Batch Prediction Mode (BONUS)

1. **Select Model**: Choose appropriate dataset model from sidebar
2. **Navigate to Tab**: Click "Batch Predictions" tab
3. **View Format**: Expand "View Expected CSV Format" to see required columns
4. **Upload CSV**: Drag & drop or browse to upload your CSV file
5. **Preview**: Review data preview to ensure correct format
6. **Process**: Click "Process All Predictions" button
7. **View Results**: See summary statistics, risk distribution, and detailed results table
8. **Download**: Click "Download Results as CSV" to save predictions

#### CSV Format Requirements

**Dataset 1 (Internet Service):**

```csv
is_tv_subscriber,is_movie_package_subscriber,subscription_age,bill_avg,remaining_contract,service_failure_count,download_avg,upload_avg,download_over_limit
1,1,12.5,45,3.0,0,50.0,5.0,0
0,1,8.0,30,0.0,2,75.0,3.0,1
```

**Dataset 2 (Telco Customer):**

```csv
gender,senior_citizen,partner,dependents,tenure,phone_service,paperless_billing,monthly_charges,total_charges
1,0,1,1,24,1,1,65.50,1572.00
0,1,0,0,6,1,0,45.00,270.00
```

**Sample test files are included** in the project for testing batch predictions.

---

## **Performance & Scalability**

### Latency (Response Time)

#### Model Loading

- **First Load**: 1-2 seconds
- **Cached**: <100ms (after initial load)
- Models loaded once at startup and cached in memory

#### Single Prediction

- **Total Time**: 50-150ms per prediction
  - Feature engineering: ~10ms
  - Scaling: ~20ms
  - Model prediction: ~20-50ms
  - UI rendering: ~50ms

#### Batch Prediction

- **Processing Speed**: ~10-20 predictions per second
- **100 customers**: ~5-10 seconds
- **1000 customers**: ~50-100 seconds
- Progress bar provides real-time feedback

### Memory Usage

#### Base Application

- **Streamlit Runtime**: ~150-200MB
- **Model Files in Memory**:
  - Dataset 1: ~320KB (XGBoost) + ~1KB (Scaler) = ~321KB
  - Dataset 2: ~305KB (XGBoost) + ~2KB (Scaler) = ~307KB
- **Total Base**: ~200-250MB

#### Per User Session

- **Session State**: ~5-10KB per active prediction
- **Batch Processing**: Additional ~1-5MB for large CSV files (1000+ rows)
- **Estimated per User**: ~10KB-5MB depending on operation

#### Memory Scaling

- **10 concurrent users**: ~250MB + (10 × 10KB) = ~250MB
- **100 concurrent users**: ~250MB + (100 × 10KB) = ~251MB
- **Batch users**: Add ~5MB per concurrent batch operation
- **Memory growth**: Minimal, mostly linear with active sessions

### Scalability Limits

#### Single Instance Performance

- **Concurrent Users**: 50-100 users (comfortable)
- **Requests per Second**: 10-20 RPS for single predictions
- **Batch Processing**: 2-5 concurrent batch jobs recommended
- **Bottleneck**: Streamlit's single-threaded architecture

#### Recommendations by Usage

**< 50 users:**

- Single instance sufficient
- 2GB RAM
- Standard setup

**50-200 users:**

- 2-3 instances behind load balancer
- 2-4GB RAM per instance
- Auto-scaling based on CPU (>70%)
- Use Docker + nginx for load balancing

**> 200 users:**

- Consider API-based architecture (FastAPI + separate frontend)
- Kubernetes for orchestration
- Separate model serving layer
- Redis caching for repeated predictions

### Optimization Notes

**Current Implementation:**

- Models loaded once at startup (efficient)
- Minimal feature engineering overhead
- Direct numpy operations (fast)
- Progress indicators for batch operations

---

## **Technical Details**

The training process for `Assignment4` concluded 63 futures and that was the cause of new pretraining of the algorithm with best parametrical results.

### Dataset 1 Features (17 total)

#### Original Features (9)

- `is_tv_subscriber`: Binary (0/1)
- `is_movie_package_subscriber`: Binary (0/1)
- `subscription_age`: Months (0-20)
- `bill_avg`: Dollars (0-100)
- `remaining_contract`: Months (0-10)
- `service_failure_count`: Count (0-5)
- `download_avg`: GB (0-150)
- `upload_avg`: GB (0-20)
- `download_over_limit`: Binary (0/1)

#### Engineered Features (8)

- `usage_ratio`: download_avg / (upload_avg + 1)
- `contract_health`: remaining_contract / (subscription_age + 1)
- `revenue_per_month`: bill_avg / (subscription_age + 1)
- `has_service_issues`: Binary (service_failures > 0)
- `log_download`: log(download_avg + 1)
- `total_services`: is_tv + is_movie
- `is_high_value`: Binary (bill_avg > 15 AND total_services > 0)
- `at_risk_flag`: Binary (remaining_contract == 0 AND service_failures > 0)

#### Model Performance

- **F1-Score**: 94.3%
- **Recall**: 93.2%
- **Precision**: 95.4%

---

### Dataset 2 Features (35 total after padding)

#### Base Features (9)

- `gender`: Binary (0=Female, 1=Male)
- `senior_citizen`: Binary (0/1)
- `partner`: Binary (0/1)
- `dependents`: Binary (0/1)
- `tenure`: Months (0-72)
- `phone_service`: Binary (0/1)
- `paperless_billing`: Binary (0/1)
- `monthly_charges`: Dollars (0-120)
- `total_charges`: Dollars (0-9000)

#### Engineered Features (4)

- `revenue_per_month`: total_charges / (tenure + 1)
- `is_new_customer`: Binary (tenure <= 12)
- `is_long_term`: Binary (tenure > 36)
- `senior_without_partner`: Binary (senior AND NOT partner)

#### Additional Features (22)

- Zero-padded to match training dimensions (one-hot encoded categorical features from original training)

#### Model Performance

- **F1-Score**: 62.8%
- **Recall**: 76.9%
- **Precision**: 53.1%

---

### Model Architecture

**Algorithm**: XGBoost Classifier

**Hyperparameters**:

```python
{
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'min_child_weight': 5,
    'gamma': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
```

**Preprocessing**: StandardScaler normalization

---

## Troubleshooting

### Models Not Found Error

**Error**: `Model files not found`

**Solution**:

1. Verify directory structure:

```
project/
├── churn.py
└── models/
    ├── xgboost_model_dataset1.pkl
    ├── scaler_dataset1.pkl
    ├── xgboost_model_dataset2.pkl
    └── scaler_dataset2.pkl
```

2. The application checks these locations in order:
   - `../models/` (parent directory)
   - `models/` (same directory)
   - `./` (current directory)

3. Ensure model files are not in `.gitignore`

---

### Version Warning Messages

**Warning**: `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.3.2 when using version 1.4.0`

**Impact**: Non-critical - predictions still work correctly

**Solution** (optional):

```bash
pip install scikit-learn==1.3.2
```

---

### Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**:

```bash
# Use different port
streamlit run churn.py --server.port=8502

# Or kill existing process
# Linux/Mac
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

### CSV Upload Errors

**Error**: `Missing required columns: [column names]`

**Solution**:

- Ensure CSV has ALL required columns with exact spelling (case-sensitive)
- Check column order doesn't matter, but all must be present
- Use provided test CSV files as templates

**Error**: `Error reading CSV file`

**Solution**:

- Ensure file is saved as CSV format (not Excel .xlsx)
- Use UTF-8 encoding
- Check for special characters or malformed rows

**Error**: Processing fails during batch prediction

**Solution**:

- Check for non-numeric values in numeric columns
- Verify no NULL/empty values in required fields
- Ensure data values are in valid ranges (e.g., no negative ages)

---

## Testing the Application

### Manual Testing

1. **Single Prediction Test**
   - Fill form with sample data
   - Verify prediction appears
   - Check recommendations are relevant
   - Test "Clear Results" button

2. **Batch Prediction Test**
   - Upload `test_dataset1_valid.csv`
   - Verify 10 predictions complete
   - Check summary statistics
   - Download results CSV

3. **Validation Test**
   - Enter invalid values (negative numbers)
   - Verify error messages appear
   - Confirm prediction button is disabled

4. **Error Handling Test**
   - Upload invalid CSV (missing columns)
   - Verify appropriate error message
   - Upload CSV with bad values
   - Check error handling

### Automated Testing

Use provided test CSV files:

- `test_dataset1_valid.csv` - Should process successfully
- `test_dataset1_invalid_missing_columns.csv` - Should show missing column error
- `test_dataset1_invalid_bad_values.csv` - Should handle gracefully

---

## **Security Considerations**

### Production Deployment

1. **Environment Variables**: Use for sensitive configurations
2. **HTTPS**: Always use TLS in production (Streamlit Cloud provides this)
3. **Input Sanitization**: Already implemented in validation functions
4. **Rate Limiting**: Implement at load balancer level
5. **Authentication**: Add user authentication for production use

### Data Privacy

- No data is stored permanently by the application
- Session data is cleared on browser close
- Models don't log individual predictions
- Consider GDPR compliance for EU users
- Batch uploads are processed in-memory only
