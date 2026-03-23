import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import joblib
import os
from datetime import datetime

# إعداد الصفحة
st.set_page_config(
    page_title="AI Prediction App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للتصميم
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        animation: fadeIn 0.5s;
    }
    .prediction-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# عنوان التطبيق مع أيقونة
st.markdown('<div class="main-header"><h1>🤖 AI Prediction App</h1><p>Smart Decision Support System</p></div>', unsafe_allow_html=True)

# Sidebar للمعلومات والإعدادات
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.markdown("## ⚙️ Settings")
    
    # اختيار النموذج
    model_type = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Random Forest", "SVM"],
        help="Choose the machine learning algorithm"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Data Statistics")
    st.markdown("""
    - **Training Samples:** 2
    - **Features:** Age, Salary
    - **Target:** Purchase Decision
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app predicts whether a customer will purchase a product based on:
    - Age
    - Salary
    
    **How to use:**
    1. Adjust age and salary
    2. Click Predict
    3. See the result
    """)

# العمودين الرئيسيين
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Input Parameters")
    
    # إنشاء tabs للإدخال
    tab1, tab2 = st.tabs(["📊 Manual Input", "📈 Batch Input"])
    
    with tab1:
        # تحسين الإدخالات باستخدام columns
        col_age, col_salary = st.columns(2)
        
        with col_age:
            age = st.slider(
                "Age",
                min_value=18,
                max_value=60,
                value=30,
                step=1,
                help="Customer's age (18-60 years)"
            )
        
        with col_salary:
            salary = st.slider(
                "Salary ($)",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=100,
                help="Customer's annual salary in USD"
            )
    
    with tab2:
        st.markdown("### Upload multiple records")
        uploaded_file = st.file_uploader(
            "Upload CSV file with columns: age, salary",
            type=['csv', 'xlsx'],
            help="Upload a file with multiple records for batch prediction"
        )
        
        if uploaded_file:
            try:
                df_batch = pd.read_csv(uploaded_file)
                if all(col in df_batch.columns for col in ['age', 'salary']):
                    st.success(f"✅ Loaded {len(df_batch)} records")
                else:
                    st.error("File must contain 'age' and 'salary' columns")
            except Exception as e:
                st.error(f"Error reading file: {e}")

with col2:
    st.markdown("### 📈 Data Visualization")
    
    # عرض بيانات التدريب بشكل تفاعلي
    training_data = pd.DataFrame({
        'Age': [25, 40],
        'Salary': [3000, 8000],
        'Purchase': ['No', 'Yes']
    })
    
    fig = px.scatter(
        training_data,
        x='Age',
        y='Salary',
        color='Purchase',
        size=[500, 500],
        title="Training Data Distribution",
        color_discrete_map={'Yes': '#28a745', 'No': '#dc3545'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# دالة لتدريب النموذج المحسن
@st.cache_resource
def train_model(model_choice):
    """تدريب النموذج مع التخزين المؤقت لتحسين الأداء"""
    X = np.array([[25, 3000], [40, 8000]])
    y = np.array([0, 1])
    
    # تطبيع البيانات
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # اختيار النموذج
    if model_choice == "Logistic Regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_choice == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
    else:
        from sklearn.svm import SVC
        model = SVC(random_state=42, probability=True)
    
    model.fit(X_scaled, y)
    
    return model, scaler

# زر التنبؤ
st.markdown("---")
col_button, col_info = st.columns([1, 2])

with col_button:
    predict_button = st.button(
        "🔮 Make Prediction",
        type="primary",
        use_container_width=True
    )

with col_info:
    st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Adjust the sliders to see how the prediction changes</div>', unsafe_allow_html=True)

# التنبؤ
if predict_button:
    with st.spinner('Analyzing data...'):
        # تدريب النموذج
        model, scaler = train_model(model_type)
        
        # تطبيع بيانات الإدخال
        input_data = np.array([[age, salary]])
        input_scaled = scaler.transform(input_data)
        
        # التنبؤ
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # عرض النتيجة مع تحسينات
        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        if prediction[0] == 1:
            with col_result1:
                st.markdown("""
                <div class="prediction-success">
                <h3>✅ Will Buy</h3>
                <p>High probability of purchase</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col_result1:
                st.markdown("""
                <div class="prediction-error">
                <h3>❌ Will Not Buy</h3>
                <p>Low probability of purchase</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            st.metric(
                label="Confidence Score",
                value=f"{prediction_proba[0][prediction[0]]:.1%}",
                delta=f"{'High' if prediction_proba[0][prediction[0]] > 0.7 else 'Medium' if prediction_proba[0][prediction[0]] > 0.5 else 'Low'}"
            )
        
        with col_result3:
            st.metric(
                label="Recommendation",
                value="✅ Proceed" if prediction[0] == 1 else "⚠️ Review",
                delta="Based on model prediction"
            )
        
        # عرض تفاصيل إضافية
        with st.expander("📊 Detailed Analysis"):
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("**Input Features:**")
                st.write(f"- Age: {age} years")
                st.write(f"- Salary: ${salary:,}")
            
            with col_detail2:
                st.markdown("**Probability Distribution:**")
                proba_df = pd.DataFrame({
                    'Decision': ['Will Not Buy', 'Will Buy'],
                    'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
                })
                
                fig_proba = px.bar(
                    proba_df,
                    x='Decision',
                    y='Probability',
                    color='Decision',
                    color_discrete_map={'Will Buy': '#28a745', 'Will Not Buy': '#dc3545'},
                    text='Probability'
                )
                fig_proba.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                fig_proba.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_proba, use_container_width=True)
        
        # معالجة الدفعات إذا كان هناك ملف مرفوع
        if 'uploaded_file' in locals() and uploaded_file:
            st.markdown("---")
            st.markdown("## 📊 Batch Predictions")
            
            # تطبيع بيانات الدفعة
            batch_features = df_batch[['age', 'salary']].values
            batch_scaled = scaler.transform(batch_features)
            batch_predictions = model.predict(batch_scaled)
            batch_proba = model.predict_proba(batch_scaled)
            
            # إضافة النتائج إلى DataFrame
            df_batch['Prediction'] = ['Will Buy' if pred == 1 else 'Will Not Buy' for pred in batch_predictions]
            df_batch['Confidence'] = [proba[pred] for proba, pred in zip(batch_proba, batch_predictions)]
            df_batch['Confidence'] = df_batch['Confidence'].apply(lambda x: f"{x:.1%}")
            
            # عرض النتائج
            st.dataframe(
                df_batch,
                use_container_width=True,
                column_config={
                    "age": st.column_config.NumberColumn("Age", format="%d"),
                    "salary": st.column_config.NumberColumn("Salary", format="$%d"),
                    "Prediction": st.column_config.TextColumn("Prediction"),
                    "Confidence": st.column_config.TextColumn("Confidence")
                }
            )
            
            # إحصائيات الدفعة
            st.markdown("### 📈 Batch Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total Records", len(df_batch))
            with col_stat2:
                st.metric("Will Buy", (df_batch['Prediction'] == 'Will Buy').sum())
            with col_stat3:
                st.metric("Will Not Buy", (df_batch['Prediction'] == 'Will Not Buy').sum())

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit | AI Prediction Model v2.0"
    "</div>",
    unsafe_allow_html=True
)