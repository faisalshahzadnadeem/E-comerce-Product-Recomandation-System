import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Function to load model and scaler
def load_and_preprocess_data():
    train_data = pd.read_csv("train.csv", engine='python', on_bad_lines='skip')
    train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce')
    train_data = train_data.dropna()

    # Feature Engineering
    train_data['Offer %'] = train_data['Offer %'].replace('%', '', regex=True).astype(float) / 100
    train_data['price1'] = pd.to_numeric(train_data['price1'], errors='coerce')
    train_data['actprice1'] = pd.to_numeric(train_data['actprice1'], errors='coerce')

    features = ['Rating', 'price1', 'actprice1', 'Offer %', 'norating1', 'noreviews1', 'star_5f', 'star_4f', 'star_3f', 'star_2f', 'star_1f']
    target = 'fulfilled1'

    X = train_data[features]
    y = train_data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, train_data

# Function to plot data
def plot_data(train_data):
    st.subheader("Data Overview")
    st.write(train_data.head())

    # Filter only numeric columns for correlation matrix
    numeric_data = train_data.select_dtypes(include=[np.number])

    # Histogram of Ratings
    plt.figure(figsize=(10, 5))
    sns.histplot(train_data['Rating'], bins=20, kde=True)
    plt.title('Distribution of Ratings')
    st.pyplot(plt.gcf())

    # Price vs Rating Scatter Plot
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='price1', y='Rating', data=train_data)
    plt.title('Price vs Rating')
    st.pyplot(plt.gcf())

    # Correlation Matrix Heatmap
    if not numeric_data.empty:
        plt.figure(figsize=(12, 8))
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        st.pyplot(plt.gcf())
    else:
        st.write("No numeric data available for correlation matrix.")

# Main function for Streamlit app
def main():
    st.title("Recommender System for E-commerce")

    # Load model, scaler, and data
    model, scaler, train_data = load_and_preprocess_data()

    # Input fields
    rating = st.number_input('Rating', min_value=0.0, max_value=5.0, step=0.1)
    price1 = st.number_input('Price', min_value=0, step=1)
    actprice1 = st.number_input('Actual Price', min_value=0, step=1)
    offer_percent = st.number_input('Offer Percentage', min_value=0, max_value=100, step=1)
    norating1 = st.number_input('Number of Ratings', min_value=0, step=1)
    noreviews1 = st.number_input('Number of Reviews', min_value=0, step=1)
    star_5f = st.number_input('5-Star Ratings', min_value=0, step=1)
    star_4f = st.number_input('4-Star Ratings', min_value=0, step=1)
    star_3f = st.number_input('3-Star Ratings', min_value=0, step=1)
    star_2f = st.number_input('2-Star Ratings', min_value=0, step=1)
    star_1f = st.number_input('1-Star Ratings', min_value=0, step=1)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Rating': [rating],
        'price1': [price1],
        'actprice1': [actprice1],
        'Offer %': [offer_percent / 100],
        'norating1': [norating1],
        'noreviews1': [noreviews1],
        'star_5f': [star_5f],
        'star_4f': [star_4f],
        'star_3f': [star_3f],
        'star_2f': [star_2f],
        'star_1f': [star_1f]
    })

    # Predict
    if st.button('Predict'):
        try:
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            prediction_label = 'Fulfilled' if prediction[0] == 1 else 'Not Fulfilled'
            st.write(f'Prediction Result: {prediction_label}')
        except Exception as e:
            st.write(f'Error during prediction: {e}')

    # Plot data
    plot_data(train_data)

if __name__ == "__main__":
    main()




