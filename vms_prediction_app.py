import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import warnings
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import streamlit as st

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
        st.header("New Maintenance Request")

        # Initialize predictions history if not exists
        if 'predictions_history' not in st.session_state:
            st.session_state.predictions_history = []

        with st.form("prediction_form"):
            st.subheader("🚗 Quick Maintenance Request")
            st.markdown("*Just fill in what you know - we'll estimate the rest!*")
            
            # Primary inputs - what users typically know
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📝 What's the problem?**")
                description = st.text_area(
                    "Describe the issue:",
                    placeholder="Example: brake noise, flat tire, engine won't start, routine service, oil change...",
                    height=100,
                    help="Describe in your own words - English or Bahasa Malaysia"
                )
                
                odometer = st.number_input("📏 Current Mileage (KM)", 
                                           min_value=1, value=200000,
                                           help="What's showing on your odometer?")
            
            with col2:
                st.markdown("**🚗 Vehicle Info (Optional)**")
                
                # Auto-detect priority based on description
                def auto_detect_priority(desc):
                    desc_lower = desc.lower()
                    if any(word in desc_lower for word in ['emergency', 'urgent', 'breakdown', 'tidak boleh', 'rosak teruk', 'accident']):
                        return 1  # Critical
                    elif any(word in desc_lower for word in ['noise', 'problem', 'issue', 'bunyi', 'masalah']):
                        return 2  # High
                    else:
                        return 1  # Default to Critical (most common in training)
                
                # Simplified priority
                use_auto_priority = st.checkbox("🤖 Auto-detect urgency", value=True,
                                               help="Let AI determine urgency from your description")

                if not use_auto_priority:
                    priority_options = {
                        "🔴 Critical - Emergency": 1,
                        "🟠 High - Important": 2,
                        "🟡 Normal - Standard": 3,
                        "⚪ Low - Routine": 0,
                    }
                    priority_selection = st.selectbox("How urgent?", list(priority_options.keys()))
                    priority = priority_options[priority_selection]
                else:
                    priority = auto_detect_priority(description)
                    urgency_text = {1: "🔴 Critical", 2: "🟠 High", 3: "🟡 Normal", 0: "⚪ Low"}
                    st.info(f"Auto-detected: {urgency_text.get(priority, '🔴 Critical')}")

                vehicle_encoded = st.number_input("🚗 Vehicle ID (if known)",
                                                  min_value=9, max_value=1137, value=573,
                                                  help="Leave as default if unknown")
            
            # Collapsible advanced options
            with st.expander("⚙️ More Details (Optional)", expanded=False):
                st.markdown("*Only change these if you know specific details*")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Auto-estimate service count based on odometer
                    estimated_services = max(2, min(2704, odometer // 15000))  # Rough estimate
                    service_count = st.number_input("🔧 How many times serviced before?", 
                                                    min_value=2, max_value=2704, 
                                                    value=estimated_services,
                                                    help=f"We estimated {estimated_services} based on your mileage")
                    
                    building_options = {
                        "Main Workshop": 2,      # Most common
                        "Branch Workshop": 3,
                        "Service Center": 1,
                        "Repair Shop": 6,
                        "Maintenance Bay": 7,
                        "Other Location": 0,
                    }
                    building_selection = st.selectbox("📍 Where will you take it?", list(building_options.keys()))
                    building_encoded = building_options[building_selection]
                
                with col4:
                    st.markdown("**ℹ️ Additional Info**")
                    st.write("✅ Request type and status are automatically determined from your problem description.")
                    st.write("💡 This helps our AI give you the most accurate diagnosis.")
            
            # Hidden fields - auto-detected but not shown to user
            status_encoded = 3  # Default to "Completed" (99.4% of training data)
            
            # Auto-detect request type based on description
            def auto_detect_mrtype(desc):
                desc_lower = desc.lower()
                if any(word in desc_lower for word in ['service', 'servis', 'maintenance', 'check']):
                    return 0  # Most common type
                elif any(word in desc_lower for word in ['repair', 'fix', 'broken', 'rosak', 'baiki']):
                    return 1
                else:
                    return 0  # Default to most common
            
            mrtype_encoded = auto_detect_mrtype(description)
            
            # Timing (simplified)
            st.markdown("**📅 When do you need this done?**")
            col5, col6 = st.columns(2)
            
            with col5:
                request_date = st.date_input("📅 Request Date", value=date.today())
            with col6:
                response_days = st.number_input("⏰ How urgent? (days)", min_value=0, value=1,
                                                help="0 = Today, 1 = Tomorrow, etc.")
            
            # Hidden fields set to defaults
            request_time = datetime.now().time()

            # Center the submit button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("🔮 Analyze My Vehicle Problem", type="primary",
                                                  use_container_width=True)

            if submitted:
                if not description.strip():
                    st.error("⚠️ Please tell us what's wrong with your vehicle!")
                    st.stop()

                # Show simplified AI analysis (no technical details)
                detected_priority = {1: "High Priority", 2: "Medium Priority", 3: "Normal", 0: "Low Priority"}
                
                st.info("🤖 **AI Analysis:** " + 
                       f"Urgency: {detected_priority.get(priority, 'High')}, " +
                       f"Estimated {service_count} previous services")

                # Prepare data
                request_datetime = datetime.combine(request_date, request_time)
                request_data = {
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

                # Make prediction
                with st.spinner("🤖 Analyzing your maintenance request..."):
                    result = st.session_state.prediction_service.predict(request_data, return_probabilities=True)

                if result:
                    # Get prediction details
                    category = result['predictions'][0]
                    confidence_level = result['confidence_levels'][0] if 'confidence_levels' in result else 'Unknown'
                    confidence_score = result['confidence_scores'][0] if 'confidence_scores' in result else 0

                    st.markdown("---")
                    st.subheader("🎯 Your Vehicle Diagnosis")

                    # Simple, clear result box
                    if category == 'other':
                        solution = "🔧 General Repair Needed"
                        action = "Take to workshop for diagnosis and repair"
                        time_needed = "1-2 days"
                        cost_estimate = "RM 200 - RM 800"
                    elif category == 'cleaning':
                        solution = "🧽 Vehicle Cleaning Service"
                        action = "Schedule vehicle washing and cleaning"
                        time_needed = "Same day"
                        cost_estimate = "RM 50 - RM 150"
                    elif category == 'tire':
                        solution = "🛞 Tire Service Required"
                        action = "Check/replace tires, wheel alignment"
                        time_needed = "Same day"
                        cost_estimate = "RM 100 - RM 600"
                    elif category == 'service':
                        solution = "⚙️ Routine Maintenance"
                        action = "Schedule regular service (oil, filters, check-up)"
                        time_needed = "Half day"
                        cost_estimate = "RM 200 - RM 500"
                    elif category == 'engine':
                        solution = "🚗 Engine Repair"
                        action = "Engine diagnosis and repair needed"
                        time_needed = "1-3 days"
                        cost_estimate = "RM 400 - RM 2000"
                    elif category == 'mechanical':
                        solution = "🔧 Mechanical Repair"
                        action = "Mechanical parts need repair/replacement"
                        time_needed = "1-2 days"
                        cost_estimate = "RM 300 - RM 1200"
                    elif category == 'brake_system':
                        solution = "🛑 Brake System Service"
                        action = "URGENT: Brake inspection and repair"
                        time_needed = "Same day"
                        cost_estimate = "RM 200 - RM 800"
                    elif category == 'hydraulic':
                        solution = "💧 Hydraulic System Repair"
                        action = "Check hydraulic fluid and system"
                        time_needed = "1-2 days"
                        cost_estimate = "RM 300 - RM 1000"
                    elif category == 'air_system':
                        solution = "💨 Air System Service"
                        action = "Air brake/suspension system check"
                        time_needed = "1 day"
                        cost_estimate = "RM 250 - RM 700"
                    elif category == 'electrical':
                        solution = "⚡ Electrical System Repair"
                        action = "Check wiring, battery, electrical components"
                        time_needed = "Half day to 1 day"
                        cost_estimate = "RM 150 - RM 600"
                    elif category == 'body':
                        solution = "🚛 Body Work Required"
                        action = "Vehicle body repair or maintenance"
                        time_needed = "1-3 days"
                        cost_estimate = "RM 300 - RM 1500"
                    else:
                        solution = "🔍 Need Further Inspection"
                        action = "Take to workshop for detailed diagnosis"
                        time_needed = "1 day"
                        cost_estimate = "RM 200 - RM 600"

                    # Main result card
                    st.markdown(f"""
                    <div style="
                        background-color: #e8f4f8;
                        border: 3px solid #2196F3;
                        border-radius: 15px;
                        padding: 25px;
                        margin: 20px 0;
                        text-align: center;
                    ">
                        <h2 style="margin: 0; color: #1976D2;">{solution}</h2>
                        <h3 style="margin: 10px 0; color: #424242;">{action}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Key information in simple cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #f57c00; margin: 0;">⏰ Time Needed</h3>
                            <h2 style="color: #e65100; margin: 10px 0;">{time_needed}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #388e3c; margin: 0;">💰 Estimated Cost</h3>
                            <h2 style="color: #2e7d32; margin: 10px 0;">{cost_estimate}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Calculate next service
                        next_service_km = ((odometer // 10000) + 1) * 10000
                        km_until_service = next_service_km - odometer
                        
                        st.markdown(f"""
                        <div style="background-color: #f3e5f5; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3 style="color: #7b1fa2; margin: 0;">🔧 Next Service</h3>
                            <h2 style="color: #6a1b9a; margin: 10px 0;">{km_until_service:,} KM</h2>
                            <p style="margin: 0; color: #8e24aa;">At {next_service_km:,} KM</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Quick recommendations based on the issue
                    st.markdown("### 💡 What You Should Do:")
                    
                    if category in ['brake_system', 'engine'] or 'emergency' in description.lower():
                        st.error("🚨 **URGENT**: This is a safety issue. Get it fixed immediately before driving!")
                    elif category in ['tire', 'mechanical']:
                        st.warning("⚠️ **IMPORTANT**: Schedule repair soon to avoid further damage.")
                    else:
                        st.info("✅ **ROUTINE**: Can be scheduled at your convenience.")

                    # Simple next steps
                    st.markdown("### 📋 Next Steps:")
                    if category == 'cleaning':
                        st.write("1. 🚿 Go to vehicle wash bay")
                        st.write("2. 🧽 Request interior and exterior cleaning")
                        st.write("3. ✅ Inspect vehicle after cleaning")
                    elif category in ['brake_system', 'engine']:
                        st.write("1. 🚨 Stop driving if unsafe")
                        st.write("2. 📞 Call workshop immediately")
                        st.write("3. 🔧 Schedule emergency repair")
                    elif category == 'tire':
                        st.write("1. 🛞 Check tire condition and pressure")
                        st.write("2. 🔧 Replace if damaged or worn")
                        st.write("3. ⚖️ Check wheel alignment")
                    else:
                        st.write("1. 📞 Call workshop to book appointment")
                        st.write("2. 🚗 Bring vehicle for inspection")
                        st.write("3. ✅ Follow mechanic's recommendations")

                    # Show confidence if high
                    if confidence_score >= 0.7:
                        st.success(f"🎯 **AI is {confidence_score:.0%} confident** about this diagnosis")
                    else:
                        st.warning(f"🤔 **AI is {confidence_score:.0%} confident** - recommend getting a second opinion")

                    # Add to history with simple format
                    history_entry = {
                        'Time': datetime.now().strftime("%H:%M"),
                        'Vehicle': vehicle_encoded,  
                        'Issue': solution,
                        'Cost': cost_estimate,
                        'Time Needed': time_needed
                    }
                    st.session_state.predictions_history.append(history_entry)
                    
                    st.success("✅ Diagnosis saved to your history!")
                else:
                    st.error("❌ Sorry, couldn't analyze your vehicle problem. Please try again with more details.")

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
            'Priority': [1, 2, 0],
            'service_count': [500, 800, 300],
            'Building_encoded': [2, 3, 1],
            'Vehicle_encoded': [573, 750, 400],
            'Status_encoded': [3, 3, 3],
            'MrType_encoded': [0, 1, 2],
            'request_date': ['2024-01-15 10:30:00', '2024-01-16 14:00:00', '2024-01-17 09:15:00'],
            'response_days': [1, 1, 2],
            'Odometer': [200000, 350000, 150000],
            'Description': [
                'adjust brake',
                'tayar pancit - flat tire needs replacement',
                'cuci lori - routine cleaning service'
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
