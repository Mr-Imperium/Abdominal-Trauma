import streamlit as st
import numpy as np
import pydicom
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import pandas as pd

# Injury columns
injury_cols = ['bowel_injury', 'extravasation_injury', 
               'kidney_healthy', 'kidney_low', 'kidney_high',
               'liver_healthy', 'liver_low', 'liver_high',
               'spleen_healthy', 'spleen_low', 'spleen_high']

# Function to load and preprocess image
def load_and_preprocess_image(uploaded_file, target_size=(224, 224)):
    # Check file type and read accordingly
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Read image based on file type
    if file_extension == 'dcm':
        # DICOM file handling
        dicom = pydicom.dcmread(uploaded_file)
        img = dicom.pixel_array
        
        # Normalize DICOM image
        if img.max() != 0:
            img = img / img.max()
    else:
        # For standard image formats (jpg, png, etc.)
        img = np.array(PIL.Image.open(uploaded_file))
        
        # Convert to grayscale if needed
        if len(img.shape) == 3 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Normalize standard image
        img = img.astype('float32') / 255.0
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Convert to 3 channels
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Streamlit app
def main():
    st.set_page_config(page_title="Abdominal Trauma Detection", layout="wide")
    
    st.title('🏥 Abdominal Trauma Detection System')
    st.write('Upload a medical image (DICOM, JPG, PNG) for injury prediction')

    # Load pre-trained model
    @st.cache_resource
    def load_trained_model():
        try:
            return load_model('best_model.keras')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    model = load_trained_model()

    # File uploader with multiple file types
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["dcm", "jpg", "jpeg", "png", "bmp", "tiff"]
    )

    if uploaded_file is not None and model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the uploaded image
            st.subheader("Uploaded Image")
            
            # Handle different file types for display
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'dcm':
                dicom_img = pydicom.dcmread(uploaded_file)
                plt.figure(figsize=(8, 8))
                plt.imshow(dicom_img.pixel_array, cmap='gray')
                plt.axis('off')
                st.pyplot(plt)
            else:
                # For standard image formats
                st.image(uploaded_file, use_container_width=True)

        with col2:
            # Preprocess the image
            try:
                processed_img = load_and_preprocess_image(uploaded_file)

                # Make prediction
                prediction = model.predict(processed_img)

                # Display predictions
                st.subheader('Injury Predictions')
                
                # Create a DataFrame for better visualization
                pred_df = pd.DataFrame({
                    'Injury Type': injury_cols,
                    'Probability': prediction[0].astype(float)  # Ensure float type
                })
                pred_df = pred_df.sort_values('Probability', ascending=False)
                
                # Custom color coding function
                def color_risk(row):
                    # Convert row to series if it's not already
                    if not isinstance(row, pd.Series):
                        row = pd.Series(row)
                    
                    # Get the injury type
                    injury_type = row.get('Injury Type', '')
                    prob = float(row.get('Probability', 0))
                    
                    # Healthy organs always in green
                    if 'healthy' in str(injury_type):
                        return pd.Series(['background-color: lightgreen'] * 2, index=['Injury Type', 'Probability'])
                    
                    # Other injury types color coding
                    if prob > 0.7:
                        return pd.Series(['background-color: darkred; color: white'] * 2, index=['Injury Type', 'Probability'])
                    elif prob > 0.5:
                        return pd.Series(['background-color: orange; color: black'] * 2, index=['Injury Type', 'Probability'])
                    elif prob > 0.3:
                        return pd.Series(['background-color: yellow; color: black'] * 2, index=['Injury Type', 'Probability'])
                    else:
                        return pd.Series(['background-color: white; color: black'] * 2, index=['Injury Type', 'Probability'])

                # Apply styling
                styled_df = pred_df.style.apply(color_risk, axis=1)
                
                # Convert to HTML for Streamlit display
                styled_html = styled_df.to_html()
                st.write(styled_html, unsafe_allow_html=True)

                # Visualization of predictions
                plt.figure(figsize=(12, 6))
                
                # Custom color mapping
                def get_bar_color(injury_type, prob):
                    if 'healthy' in injury_type:
                        return 'green'
                    elif prob > 0.7:
                        return 'red'
                    elif prob > 0.5:
                        return 'orange'
                    elif prob > 0.3:
                        return 'yellow'
                    else:
                        return 'lightgray'

                # Plot bars with custom colors
                bars = plt.bar(
                    pred_df['Injury Type'], 
                    pred_df['Probability'],
                    color=[get_bar_color(type, prob) for type, prob in zip(pred_df['Injury Type'], pred_df['Probability'])]
                )
                plt.title('Injury Probabilities', fontsize=16)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1)  # Set y-axis from 0 to 1
                plt.tight_layout()
                st.pyplot(plt)

                # Refined Risk Summary
                st.subheader('Risk Summary')
                # Exclude healthy organs from risk summary
                risk_df = pred_df[~pred_df['Injury Type'].str.contains('healthy')]
                high_risk = risk_df[risk_df['Probability'].astype(float) > 0.5]
                
                if not high_risk.empty:
                    st.warning("High-Risk Injuries Detected:")
                    for _, row in high_risk.iterrows():
                        st.error(f"{row['Injury Type']}: {float(row['Probability']):.2f}")
                else:
                    st.success("No high-risk injuries detected.")

            except Exception as e:
                st.error(f"Error processing image: {e}")
                # Print full error for debugging
                import traceback
                st.error(traceback.format_exc())

    # Additional information
    st.sidebar.title("About the Model")
    st.sidebar.info("""
    - Multi-label injury detection model
    - Trained on CT scan images
    - Detects various abdominal injuries
    - Model uses ResNet50 architecture
    - Best performing model used for predictions
    """)

if __name__ == '__main__':
    main()
