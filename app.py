
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the saved model

def load_model():
    # Load the model using joblib (from .pkl file)
    model = joblib.load("xgboost_model.pkl")
    return model

def predict_delivery_time(model, total_price, total_freight, total_payment, avg_review_score, 
                          purchase_hour, purchase_day, purchase_month, purchase_weekday, 
                          customer_state, seller_state, product_category, payment_type):
    """Prepare input data and predict the estimated delivery time."""
    input_data = pd.DataFrame([[total_price, total_freight, total_payment, avg_review_score, 
                                purchase_hour, purchase_day, purchase_month, purchase_weekday, 
                                customer_state, seller_state, product_category, payment_type]], 
                               columns=["total_price", "total_freight", "total_payment", "avg_review_score", 
                                        "purchase_hour", "purchase_day", "purchase_month", "purchase_weekday", 
                                        "customer_state", "seller_state", "product_category_name_english", "payment_type"])
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

def setup_ui():
    """Initialize and set up the Streamlit UI."""
    st.set_page_config(page_title="Order Delivery Time Prediction", layout="centered")
    st.title("🚚 Order Delivery Time Prediction")
    st.write("Predict the estimated delivery time based on order details.")
    st.sidebar.header("📌 How to Use")
    st.sidebar.write("""
    - Enter order details including product category, customer location, and shipping method.
    - Click **'Predict Delivery Time'** to get an estimated time.
    - The model will predict the time in hours.
    """)

def main():
    model = load_model()  # Load the trained model

    # Input fields for the user
    total_price = st.number_input("Total Price", min_value=0.0, step=0.01)
    total_freight = st.number_input("Total Freight", min_value=0.0, step=0.01)
    total_payment = st.number_input("Total Payment", min_value=0.0, step=0.01)
    avg_review_score = st.number_input("Average Review Score", min_value=1, max_value=5, step=1)
    purchase_hour = st.number_input("Purchase Hour", min_value=0, max_value=23, step=1)
    purchase_day = st.number_input("Purchase Day", min_value=1, max_value=31, step=1)
    purchase_month = st.number_input("Purchase Month", min_value=1, max_value=12, step=1)
    purchase_weekday = st.number_input("Purchase Weekday (0=Monday, 6=Sunday)", min_value=0, max_value=6, step=1)
    customer_state = st.selectbox("Customer State", ["SP", "RJ", "MG", "RS", "PR"])  # Add actual states
    seller_state = st.selectbox("Seller State", ["SP", "RJ", "MG", "RS", "PR"])  # Add actual states
    product_category = st.selectbox("Product Category", ["electronics", "clothing", "furniture"])  # Update categories
    payment_type = st.selectbox("Payment Type", ["credit_card", "boleto", "voucher", "debit_card"])

    # Prediction button
    if st.button("Predict Delivery Time"):
        delivery_time = predict_delivery_time(
            model, total_price, total_freight, total_payment, avg_review_score,
            purchase_hour, purchase_day, purchase_month, purchase_weekday,
            customer_state, seller_state, product_category, payment_type
        )
        st.success(f"Predicted Delivery Time: {delivery_time} hours")

if __name__ == "__main__":
    setup_ui()
    main()
