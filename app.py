import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_vehicles1.csv')
    df = df.drop(['id', 'region', 'model', 'description', 'image_url'], axis=1, errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# Function to train and evaluate model
def train_model(df):
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify columns
    categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 
                        'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state']
    numerical_cols = ['year', 'odometer', 'car_age']  # Removed price_per_mile to avoid data leakage
    
    # Preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # XGBoost model with optimal parameters
    xgb_model = XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    # Create pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb_model)
    ])
    
    # Fit the model
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline, X_train, X_test, y_train, y_test

# Main Streamlit app
def main():
    st.set_page_config(page_title="Car Price Predictor", layout="centered")
    st.title('🚗 Car Price Predictor')
    
    # Sidebar for model training
    st.sidebar.header("Model Training Options")
    if st.sidebar.button("Train New Model"):
        with st.spinner("Training model... This may take a few minutes"):
            df = load_data()
            model_pipeline, X_train, X_test, y_train, y_test = train_model(df)
            joblib.dump(model_pipeline, 'car_price_predictor.pkl')
            st.sidebar.success("Model trained and saved successfully!")
    
    # Load trained model
    try:
        model = joblib.load('car_price_predictor.pkl')
        df = load_data()
    except Exception as e:
        st.error(f"No trained model found. Please train a model first. Error: {str(e)}")
        return
    
    # Prediction form
    with st.form("car_details"):
        st.header('Enter Car Details')
        
        year = st.number_input('Year', min_value=1990, max_value=2023, value=2015)
        odometer = st.number_input('Odometer (miles)', min_value=0, max_value=500000, value=50000)
        car_age = 2023 - year
        
        manufacturer = st.selectbox('Manufacturer', df['manufacturer'].dropna().unique())
        condition = st.selectbox('Condition', df['condition'].dropna().unique())
        cylinders = st.selectbox('Cylinders', df['cylinders'].dropna().unique())
        fuel = st.selectbox('Fuel Type', df['fuel'].dropna().unique())
        title_status = st.selectbox('Title Status', df['title_status'].dropna().unique())
        transmission = st.selectbox('Transmission', df['transmission'].dropna().unique())
        drive = st.selectbox('Drive Type', df['drive'].dropna().unique())
        car_type = st.selectbox('Car Type', df['type'].dropna().unique())
        paint_color = st.selectbox('Paint Color', df['paint_color'].dropna().unique())
        state = st.selectbox('State', df['state'].dropna().unique())
        
        submitted = st.form_submit_button("Predict Price 💰")

    if submitted:
        # Create input data (without price_per_mile)
        input_df = pd.DataFrame({
            'year': [year],
            'odometer': [odometer],
            'car_age': [car_age],
            'manufacturer': [manufacturer],
            'condition': [condition],
            'cylinders': [cylinders],
            'fuel': [fuel],
            'title_status': [title_status],
            'transmission': [transmission],
            'drive': [drive],
            'type': [car_type],
            'paint_color': [paint_color],
            'state': [state]
        })

        try:
            # Get prediction (in dollars)
            predicted_price_dollars = model.predict(input_df)[0]
            # Convert to rupees
            predicted_price_rupees = predicted_price_dollars
            
            st.success(f"Predicted Car Price: ₹{predicted_price_rupees:,.2f}")
            st.markdown("### Details of Your Input:")
            st.dataframe(input_df)

            # Find closest actual match
            df_check = df.copy()
            df_check['car_age'] = 2023 - df_check['year']
            
            filter_conditions = (
                (df_check['manufacturer'] == manufacturer) &
                (df_check['condition'] == condition) &
                (df_check['cylinders'] == cylinders) &
                (df_check['fuel'] == fuel) &
                (df_check['title_status'] == title_status) &
                (df_check['transmission'] == transmission) &
                (df_check['drive'] == drive) &
                (df_check['type'] == car_type) &
                (df_check['paint_color'] == paint_color) &
                (df_check['state'] == state)
            )

            match = df_check[filter_conditions]
            if not match.empty:
                match['diff'] = np.abs(match['odometer'] - odometer)
                closest = match.loc[match['diff'].idxmin()]
                st.info(f"Closest Actual Price from Dataset: ₹{closest['price']:,.2f}")
                
                # Plot comparison with BOTH prices in rupees
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(['Predicted', 'Actual'], 
                             [predicted_price_rupees, closest['price']],
                             color=['#1f77b4', '#ff7f0e'])
                
                ax.set_ylabel('Price (₹)')
                ax.set_title('Price Comparison')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'₹{height:,.2f}',
                            ha='center', va='bottom')
                
                # Set dynamic y-axis limits
                max_price = max(predicted_price_rupees, closest['price'])
                ax.set_ylim(0, max_price * 1.2)  # Add 20% padding
                
                st.pyplot(fig)
            else:
                st.warning("No exact match found for your configuration in the dataset.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()