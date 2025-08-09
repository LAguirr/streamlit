import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Title of the app
st.title("Excel Data Preparation and Model Training App")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls","csv"])

if uploaded_file is not None:

    # Load the Excel file into a DataFrame
    if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        st.write("Excel file uploaded successfully!")
        df = pd.read_excel(uploaded_file)

    # If the file is a CSV, read it directly
    if uploaded_file.name.endswith('.csv'):
        st.write("CSV file uploaded successfully!")
        df = pd.read_csv(uploaded_file)

    # Display the raw data
    st.subheader("Raw Data")
    st.write(df)

    # Data Cleaning and Preprocessing
    st.subheader("Data Cleaning and Preprocessing")

    # Handle missing values
    st.write("### Handle Missing Values")
    if df.isnull().sum().sum() > 0:
        st.write("Missing values detected:")
        missing_counts = df.isnull().sum()
        st.write(missing_counts[missing_counts > 0])  # Only show columns with missing values

        # Get list of columns with missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        
        # Let user select which columns to handle
        selected_cols = st.multiselect(
            "Select columns to handle (default: all)",
            options=cols_with_missing,
            default=cols_with_missing
        )
        
        if not selected_cols:
            st.warning("Please select at least one column to proceed")
            st.stop()

        # Option to drop or fill missing values
        missing_value_option = st.radio(
            "Choose how to handle missing values:",
            ("Drop rows with missing values", "Fill missing values"),
            horizontal=True
        )

        if missing_value_option == "Drop rows with missing values":
            # Only drop rows where selected columns have missing values
            df = df.dropna(subset=selected_cols)
            st.success(f"Rows with missing values in selected columns dropped. New shape: {df.shape}")
        else:
            with st.expander("Advanced Filling Options", expanded=True):
                for col in selected_cols:
                    st.subheader(f"Column: {col}")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Check data type of column
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fill_method = st.selectbox(
                                f"Method for '{col}'",
                                ["Mean", "Median", "Specific value", "Interpolate", "KNN Impute"],
                                key=f"num_{col}"
                            )
                        else:
                            fill_method = st.selectbox(
                                f"Method for '{col}'",
                                ["Mode", "Specific value", "Forward fill", "Backward fill"],
                                key=f"cat_{col}"
                            )
                    
                    with col2:
                        if fill_method == "Specific value":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                val = st.number_input(
                                    f"Enter fill value for '{col}'",
                                    value=df[col].mean() if not df[col].empty else 0,
                                    key=f"num_val_{col}"
                                )
                            else:
                                val = st.text_input(
                                    f"Enter fill value for '{col}'",
                                    value="UNKNOWN",
                                    key=f"cat_val_{col}"
                                )
                            df[col] = df[col].fillna(val)
                        elif fill_method == "Mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif fill_method == "Median":
                            df[col] = df[col].fillna(df[col].median())
                        elif fill_method == "Mode":
                            df[col] = df[col].fillna(df[col].mode()[0])
                        elif fill_method == "Interpolate":
                            df[col] = df[col].interpolate()
                        elif fill_method == "Forward fill":
                            df[col] = df[col].ffill()
                        elif fill_method == "Backward fill":
                            df[col] = df[col].bfill()
                        elif fill_method == "KNN Impute":
                            try:
                                from sklearn.impute import KNNImputer
                                imputer = KNNImputer(n_neighbors=3)
                                df[col] = imputer.fit_transform(df[[col]])
                                st.info("Used KNN imputation (scikit-learn required)")
                            except ImportError:
                                st.error("KNN imputation requires scikit-learn. Using mean instead.")
                                df[col] = df[col].fillna(df[col].mean())
                        
                        st.info(f"Applied {fill_method} to column '{col}'")
                        st.write(f"Missing before: {missing_counts[col]}, After: {df[col].isnull().sum()}")

        st.write("### Data after handling missing values:")
        st.dataframe(df.style.highlight_null(color='yellow'), use_container_width=True)
    else:
        st.success("No missing values found in the dataset.")

    # Encode categorical variables
    st.write("### Encode Categorical Variables")
    categorical_columns = df.select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        st.write("Categorical columns detected:")
        st.write(categorical_columns)

        # Option to encode categorical variables
        if st.checkbox("Encode categorical variables using Label Encoding"):
            label_encoder = LabelEncoder()
            for col in categorical_columns:
                df[col] = label_encoder.fit_transform(df[col])
            st.write("Categorical variables encoded.")
            st.write(df)
    else:
        st.write("No categorical columns found.")

    # Display final cleaned data
    st.subheader("Final Cleaned Data")
    st.write(df)

    # Option to download the cleaned data
    st.write("### Download Cleaned Data")
    cleaned_file = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cleaned Data as CSV",
        data=cleaned_file,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    # Model Training Section
    st.subheader("Model Training")

    # Check if the data has a target column
    target_column = st.selectbox("Select the target column for model training:", df.columns)
    if target_column:
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split data into training and testing sets
        test_size = st.slider("Select test set size (percentage):", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Choose a model
        model_option = st.selectbox("Choose a model:", ["Random Forest Classifier"])

        if model_option == "Random Forest Classifier":
            model = RandomForestClassifier()

        # Train the model
        if st.button("Train Model"):
            st.write("Training the model...")
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display model performance
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")

            # Option to save the trained model
            if st.button("Save Model"):
                model_filename = "trained_model.pkl"
                joblib.dump(model, model_filename)
                st.write(f"Model saved as {model_filename}")

else:
    st.write("Please upload an Excel file to get started.")