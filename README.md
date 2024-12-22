# Credit Card Fraud Detection

Detect credit card fraud using Python libraries and machine learning techniques to enhance transaction security and minimize fraudulent activities.

---

## Overview
This project aims to classify fraudulent and non-fraudulent credit card transactions using machine learning algorithms. The dataset contains anonymized transaction data with features such as transaction amount, timestamp, and engineered principal components. Various Python libraries are utilized to preprocess data, build models, and visualize results.

---

## Features
- **Data Preprocessing**: 
  - Handle imbalanced datasets using techniques like SMOTE.
  - Normalize data for consistent model performance.
  - Analyze missing or anomalous data points.
- **Machine Learning Models**: 
  - Logistic Regression, Decision Tree, Random Forest, and Support Vector Machines (SVM).
- **Evaluation Metrics**: 
  - Accuracy, precision, recall, F1 score, and ROC-AUC.
- **Visualization**: 
  - Create visual insights using libraries like Matplotlib and Seaborn.

---

## Dataset
The dataset used is the Credit Card Fraud Detection Dataset, which includes:
- **Total Transactions**: 284,807
- **Fraudulent Transactions**: 492 (0.172%)
- The dataset is highly imbalanced, requiring specialized techniques for effective classification.

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

### Step 2: Navigate to the Project Directory
```bash
cd credit-card-fraud-detection
```

### Step 3: Install the Required Packages
Install the required Python libraries listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Preprocess the Dataset
Preprocess the dataset to handle missing values, normalize features, and address class imbalance:
```bash
python preprocess.py
```

### Step 2: Train the Model
Train machine learning models on the processed data:
```bash
python train.py
```

### Step 3: Evaluate the Model
Evaluate model performance using metrics like precision, recall, and ROC-AUC:
```bash
python evaluate.py
```

### Step 4: Visualize Results
Visualize the dataset and results using graphs and plots:
```bash
python visualize.py
```

---

## Python Libraries Used
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib**: For creating static plots and visualizations.
- **Seaborn**: For advanced data visualization and heatmaps.
- **Imbalanced-learn**: For handling imbalanced datasets (e.g., SMOTE).
- **Joblib**: For saving and loading trained models.

---

## Models Used
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machines (SVM)**

---

## Results
| Model                | Accuracy            | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|---------------------|-----------|--------|----------|---------|
| Logistic Regression  | 0.9985253326779256 | 0.93      | 0.91   | 0.92     | 0.98    |
| Decision Tree        | 0.996892665285629  | 0.85      | 0.87   | 0.86     | 0.95    |
| Random Forest        | 0.9985253326779256 | 0.94      | 0.92   | 0.93     | 0.99    |
| SVM                  | 0.9985253326779256 | 0.92      | 0.90   | 0.91     | 0.98    |

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements
- **Dataset**: Provided by Kaggle.
- **Inspiration**: Various open-source machine learning projects.
- **Python Libraries**: Thanks to the developers of Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn.

---

### Feel free to reach out for any queries or collaboration opportunities.
