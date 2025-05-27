import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import warnings
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="VMS Prediction Service",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


class VMSPredictionService:
    """
    Vehicle Maintenance System Prediction Service
    Makes predictions on new service requests
    """

    def __init__(self, model_path='model_training_output/maintenance_prediction_model.pkl'):
        """Initialize the prediction service"""
        self.model_path = model_path
        self.model_objects = None

    def load_model(self):
        """Load the trained model and all components"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_objects = pickle.load(f)
            return True, f"Model loaded successfully - {self.model_objects['model_type']}"
        except FileNotFoundError:
            return False, f"Model file not found at {self.model_path}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def prepare_features(self, data):
        """Prepare features from raw service request data"""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Create time-based features if request_date exists
        if 'request_date' in df.columns:
            df['request_date'] = pd.to_datetime(df['request_date'])
            df['request_day_of_week'] = df['request_date'].dt.dayofweek
            df['request_month'] = df['request_date'].dt.month
            df['request_hour'] = df['request_date'].dt.hour

        # Create enhanced features
        df_enhanced = self._create_enhanced_features(df)

        # Process features
        numerical_features = self.model_objects['numerical_features']
        X_numerical = df_enhanced[numerical_features] if numerical_features else pd.DataFrame()

        categorical_features = self.model_objects['categorical_features']
        X_categorical = df_enhanced[categorical_features] if categorical_features else pd.DataFrame()

        text_feature = self.model_objects['text_feature']
        X_text = df_enhanced[text_feature] if text_feature and text_feature in df_enhanced.columns else pd.Series('',
                                                                                                                  index=df_enhanced.index)

        return self._process_features(X_numerical, X_categorical, X_text)

    def _create_enhanced_features(self, df):
        """Create enhanced features"""
        df_enhanced = df.copy()

        if 'request_day_of_week' in df.columns:
            df_enhanced['is_weekend'] = (df['request_day_of_week'] >= 5).astype(int)

        if 'request_hour' in df.columns:
            df_enhanced['is_business_hours'] = ((df['request_hour'] >= 8) &
                                                (df['request_hour'] <= 17)).astype(int)

        if 'service_count' in df.columns:
            service_threshold = df['service_count'].quantile(0.75) if len(df) > 1 else 10
            df_enhanced['high_maintenance_vehicle'] = (df['service_count'] >= service_threshold).astype(int)

        return df_enhanced

    def _process_features(self, X_numerical, X_categorical, X_text):
        """Process features using trained preprocessors"""
        processed_features = []

        if not X_numerical.empty and self.model_objects['numerical_imputer'] is not None:
            X_num = self.model_objects['numerical_imputer'].transform(X_numerical)
            X_num = self.model_objects['numerical_scaler'].transform(X_num)
            processed_features.append(X_num)

        if not X_categorical.empty and self.model_objects['categorical_imputer'] is not None:
            X_cat = self.model_objects['categorical_imputer'].transform(X_categorical)
            processed_features.append(X_cat)

        if self.model_objects['tfidf'] is not None:
            X_text_clean = X_text.fillna('').astype(str)
            X_text_processed = self.model_objects['tfidf'].transform(X_text_clean)
            processed_features.append(X_text_processed.toarray())

        if processed_features:
            from scipy.sparse import hstack, csr_matrix
            matrices = [csr_matrix(f) for f in processed_features]
            X_combined = hstack(matrices)

            if self.model_objects['feature_selector'] is not None:
                X_final = self.model_objects['feature_selector'].transform(X_combined)
            else:
                X_final = X_combined

            return X_final
        else:
            raise ValueError("No features could be processed")

    def predict(self, data, return_probabilities=False):
        """Make predictions on new service request data"""
        if self.model_objects is None:
            return None

        try:
            X_processed = self.prepare_features(data)
            model = self.model_objects['final_model']
            predictions = model.predict(X_processed)

            label_encoder = self.model_objects['label_encoder']
            predicted_categories = label_encoder.inverse_transform(predictions)

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed)
                max_probs = np.max(probabilities, axis=1)
            else:
                probabilities = None
                max_probs = None

            results = {
                'predictions': predicted_categories.tolist(),
                'prediction_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_type': self.model_objects['model_type'],
                    'training_accuracy': self.model_objects['model_performance']['accuracy']
                }
            }

            if return_probabilities and probabilities is not None:
                results['probabilities'] = probabilities.tolist()
                results['confidence_scores'] = max_probs.tolist()

            if max_probs is not None:
                confidence_levels = []
                for prob in max_probs:
                    if prob >= 0.8:
                        confidence_levels.append('High')
                    elif prob >= 0.6:
                        confidence_levels.append('Medium')
                    else:
                        confidence_levels.append('Low')
                results['confidence_levels'] = confidence_levels

            return results

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None


# Initialize session state
if 'prediction_service' not in st.session_state:
    st.session_state.prediction_service = VMSPredictionService()
    st.session_state.model_loaded = False


def load_model():
    """Load the prediction model"""
    success, message = st.session_state.prediction_service.load_model()
    st.session_state.model_loaded = success
    return success, message


def main():
    st.title("🚗 Vehicle Maintenance System - Prediction Service")
    st.markdown("---")

    # Sidebar for model information and settings
    with st.sidebar:
        st.header("🔧 Model Settings")

        # Model loading section
        st.subheader("Load Model")
        model_path = st.text_input("Model Path", value="model_training_output/maintenance_prediction_model.pkl")

        if st.button("Load Model", type="primary"):
            st.session_state.prediction_service.model_path = model_path
            success, message = load_model()
            if success:
                st.success(message)
            else:
                st.error(message)

        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
            model_info = st.session_state.prediction_service.model_objects
            st.info(f"**Model Type:** {model_info['model_type']}")
            st.info(f"**Accuracy:** {model_info['model_performance']['accuracy']:.3f}")
        else:
            st.warning("⚠️ Model Not Loaded")

    # Main content area
    if not st.session_state.model_loaded:
        st.warning("Please load a model first using the sidebar.")
        st.info("Make sure your model file exists at the specified path.")
        return

    # Tabs for different prediction modes
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Information"])

    with tab1:
        st.header("Single Service Request Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Information")
            priority = st.selectbox("Priority", [1, 2, 3], help="Request priority level")
            service_count = st.number_input("Service Count", min_value=0, value=5, help="Number of previous services")
            building_encoded = st.number_input("Building ID", min_value=0, value=1, help="Encoded building identifier")
            vehicle_encoded = st.number_input("Vehicle ID", min_value=0, value=10, help="Encoded vehicle identifier")
            status_encoded = st.selectbox("Status", [1, 2, 3], help="Current status")
            mrtype_encoded = st.selectbox("MR Type", [1, 2, 3], help="Maintenance request type")

        with col2:
            st.subheader("Additional Details")
            request_date = st.date_input("Request Date", value=date.today())
            request_time = st.time_input("Request Time", value=time(10, 30))
            response_days = st.number_input("Expected Response Days", min_value=0, value=2)
            odometer = st.number_input("Odometer Reading", min_value=0, value=45000)
            description = st.text_area("Description", value="Engine making unusual noise, needs inspection",
                                       help="Detailed description of the maintenance issue")

        if st.button("🔮 Make Prediction", type="primary"):
            # Combine date and time
            request_datetime = datetime.combine(request_date, request_time)

            sample_request = {
                'Priority': priority,
                'service_count': service_count,
                'Building_encoded': building_encoded,
                'Vehicle_encoded': vehicle_encoded,
                'Status_encoded': status_encoded,
                'MrType_encoded': mrtype_encoded,
                'request_date': request_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'response_days': response_days,
                'Odometer': odometer,
                'Description': description
            }

            with st.spinner("Making prediction..."):
                result = st.session_state.prediction_service.predict(sample_request, return_probabilities=True)

            if result:
                st.success("Prediction completed!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicted Category", result['predictions'][0])

                with col2:
                    confidence_score = result['confidence_scores'][0] if 'confidence_scores' in result else 0
                    st.metric("Confidence Score", f"{confidence_score:.3f}")

                with col3:
                    confidence_level = result['confidence_levels'][0] if 'confidence_levels' in result else 'Unknown'
                    st.metric("Confidence Level", confidence_level)

                # Visualization of confidence
                if 'probabilities' in result:
                    probs = result['probabilities'][0]
                    classes = st.session_state.prediction_service.model_objects['classes']

                    fig = px.bar(
                        x=classes,
                        y=probs,
                        title="Prediction Probabilities",
                        labels={'x': 'Categories', 'y': 'Probability'}
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with multiple service requests for batch processing.")

        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview")
                st.dataframe(df.head())

                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        results = st.session_state.prediction_service.predict(df, return_probabilities=True)

                    if results:
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'predicted_category': results['predictions'],
                            'confidence_level': results[
                                'confidence_levels'] if 'confidence_levels' in results else ['Unknown'] * len(
                                results['predictions']),
                            'confidence_score': results[
                                'confidence_scores'] if 'confidence_scores' in results else [0.0] * len(
                                results['predictions'])
                        })

                        # Combine with original data
                        combined_df = pd.concat([df, results_df], axis=1)

                        st.subheader("Batch Prediction Results")
                        st.dataframe(combined_df)

                        # Download results
                        csv = combined_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                        # Summary statistics
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Prediction Summary")
                            category_counts = pd.Series(results['predictions']).value_counts()
                            fig_pie = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title="Distribution of Predicted Categories"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            st.subheader("Confidence Distribution")
                            if 'confidence_levels' in results:
                                conf_counts = pd.Series(results['confidence_levels']).value_counts()
                                fig_bar = px.bar(
                                    x=conf_counts.index,
                                    y=conf_counts.values,
                                    title="Confidence Level Distribution"
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

        # Sample data download
        st.subheader("Need a sample file?")
        sample_data = {
            'Priority': [1, 2, 3],
            'service_count': [5, 8, 3],
            'Building_encoded': [1, 2, 1],
            'Vehicle_encoded': [10, 15, 8],
            'Status_encoded': [2, 1, 3],
            'MrType_encoded': [3, 2, 1],
            'request_date': ['2024-01-15 10:30:00', '2024-01-16 14:00:00', '2024-01-17 09:15:00'],
            'response_days': [2, 1, 3],
            'Odometer': [45000, 67000, 23000],
            'Description': [
                'Engine making unusual noise, needs inspection',
                'Brake system maintenance required',
                'Regular maintenance check'
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Sample CSV",
            data=csv_sample,
            file_name="sample_service_requests.csv",
            mime="text/csv"
        )

    with tab3:
        st.header("Model Information & Performance")

        if st.session_state.model_loaded:
            model_info = st.session_state.prediction_service.model_objects

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Details")
                st.info(f"**Model Type:** {model_info['model_type']}")
                st.info(f"**Training Date:** {model_info['training_metadata']['training_date']}")
                st.info(f"**Training Samples:** {model_info['training_metadata']['training_samples']:,}")
                st.info(f"**Features Used:** {model_info['training_metadata']['n_features']}")

                st.subheader("Available Classes")
                for i, class_name in enumerate(model_info['classes'], 1):
                    st.write(f"{i}. {class_name}")

            with col2:
                st.subheader("Performance Metrics")

                metrics = model_info['model_performance']

                # Create gauge charts for metrics
                fig_metrics = go.Figure()

                fig_metrics.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=metrics['accuracy'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Accuracy"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))

                fig_metrics.update_layout(height=300)
                st.plotly_chart(fig_metrics, use_container_width=True)

                # Additional metrics
                st.metric("Precision", f"{metrics['precision']:.3f}")
                st.metric("Recall", f"{metrics['recall']:.3f}")
                st.metric("F1-Score", f"{metrics['f1']:.3f}")
        else:
            st.warning("Load a model to view information.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🚗 Vehicle Maintenance System Prediction Service</p>
            <p>Built with Streamlit • Powered by Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()