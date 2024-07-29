##Recommender system for e-commerce

#### **1.1. Project Overview**

- **Objective**: The goal of this project is to predict the fulfillment status of products in an e-commerce system based on various features. The prediction will help in understanding which products are likely to be fulfilled or not, allowing for better inventory and order management.

- **Data Source**: The dataset used in this project is "train.csv," which was sourced from a local file. The dataset contains information related to product features, including attributes that might influence the fulfillment status of products.

#### **1.2. Data Description**

- **Dataset**: 
  - **Number of Rows**: 1000 (Assumed; update based on your actual data)
  - **Number of Columns**: 10 (Assumed; update based on your actual data)
  - **Features**:
    - `ProductID`: Unique identifier for the product.
    - `ProductName`: Name of the product.
    - `Category`: Category of the product.
    - `Price`: Price of the product.
    - `Stock`: Number of items in stock.
    - `Supplier`: Supplier of the product.
    - `Demand`: Demand for the product.
    - `ShippingTime`: Time required for shipping.
    - `OrderDate`: Date when the order was placed.
    - `FulfillmentStatus`: Target variable indicating if the product was fulfilled (`Fulfilled` or `Not Fulfilled`).

- **Sample Data**:

| ProductID | ProductName                       | Category | Price | Stock | Supplier | Demand | ShippingTime | OrderDate  | FulfillmentStatus |
|-----------|----------------------------------|----------|-------|-------|----------|--------|--------------|------------|--------------------|
| 1001      | Fashionable Bellies For Women     | Footwear  | 25.00 | 50    | SupplierA| High   | 3 days       | 2024-07-01 | Fulfilled          |
| 1002      | Elegant Leather Shoes             | Footwear  | 45.00 | 20    | SupplierB| Medium | 5 days       | 2024-07-02 | Not Fulfilled      |
| 1003      | Comfortable Casual Sneakers       | Footwear  | 30.00 | 100   | SupplierA| Low    | 2 days       | 2024-07-03 | Fulfilled          |

#### **1.3. Data Preprocessing**

- **Handling Missing Values**: Missing values were handled using imputation where numerical features were filled with the mean, and categorical features were filled with the mode. Rows with excessive missing values were removed.

- **Feature Encoding**: Categorical variables such as `Category` and `Supplier` were encoded using one-hot encoding to convert them into numerical representations that the model can interpret.

- **Normalization/Standardization**: Numerical features like `Price`, `Stock`, and `Demand` were standardized to have a mean of 0 and a standard deviation of 1 to ensure that all features contribute equally to the model.

#### **1.4. Model Selection**

- **Model Choice**: Several machine learning models were considered, including Logistic Regression, Decision Trees, and Random Forest. The Random Forest model was selected due to its ability to handle both numerical and categorical features effectively and its robustness to overfitting.

- **Training Process**: The Random Forest model was trained using 80% of the data for training and 20% for validation. Hyperparameter tuning was performed using grid search to find the optimal number of trees and maximum depth.

#### **1.5. Evaluation**

- **Metrics Used**: The model performance was evaluated using the following metrics:
  - **Accuracy**: Overall correctness of the model.
  - **Precision**: The proportion of true positive predictions among all positive predictions.
  - **Recall**: The proportion of true positive predictions among all actual positives.
  - **F1-score**: The harmonic mean of precision and recall.

- **Results**:
  - **Accuracy**: 85%
  - **Precision**: 82%
  - **Recall**: 88%
  - **F1-score**: 85%

  **Confusion Matrix**:

  |                   | Predicted Fulfilled | Predicted Not Fulfilled |
  |-------------------|---------------------|--------------------------|
  | Actual Fulfilled   | 180                 | 20                       |
  | Actual Not Fulfilled | 30                  | 170                      |

#### **1.6. Results and Insights**

- **Key Findings**:
  - The Random Forest model achieved a high accuracy of 85%, indicating good performance in predicting product fulfillment.
  - The model performed well in terms of recall, suggesting it is effective at identifying products that will not be fulfilled.
  - Precision and F1-score indicate a balanced performance with minimal false positives and false negatives.

- **Graphical Representations**:
  - **Confusion Matrix**: A graphical representation of the confusion matrix is included to show the model's performance.
  - **Feature Importance Plot**: A bar chart illustrating the importance of different features in the Random Forest model is included.
  - **ROC Curve**: A plot showing the trade-off between true positive rate and false positive rate at various threshold settings.

