import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better spacing and readability
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        color: black;
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        gap: 10px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6e9ef;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('Cleaned_Students_Performance.csv')
    # Convert binary columns to more readable format
    df['gender'] = df['gender'].map({0: 'Female', 1: 'Male'})
    df['lunch'] = df['lunch'].map({0: 'Standard', 1: 'Free/Reduced'})
    df['test_preparation_course'] = df['test_preparation_course'].map({0: 'None', 1: 'Completed'})
    return df

# Load data
df = load_data()

# Sidebar with clear instructions
st.sidebar.title("🎯 Analysis Controls")
st.sidebar.markdown("""
    Welcome! Follow these steps to analyze the data:
    1. Select your demographic filters below
    2. Explore each tab in the main view
    3. Use advanced filters for deeper analysis
""")

# Sidebar filters with tooltips
with st.sidebar:
    with st.expander("📊 Basic Filters", expanded=True):
        st.info("These filters will affect all visualizations across the dashboard.")
        
        selected_gender = st.multiselect(
            "Select Gender",
            options=df['gender'].unique(),
            default=df['gender'].unique(),
            help="Choose one or more gender categories to analyze"
        )

        selected_ethnicity = st.multiselect(
            "Select Race/Ethnicity",
            options=df['race_ethnicity'].unique(),
            default=df['race_ethnicity'].unique(),
            help="Select specific ethnic groups to compare"
        )

        selected_parent_education = st.multiselect(
            "Select Parental Education Level",
            options=df['parental_level_of_education'].unique(),
            default=df['parental_level_of_education'].unique(),
            help="Filter by parents' education background"
        )

    with st.expander("🔍 Advanced Filters"):
        st.info("Use these filters for more specific analysis.")
        
        score_threshold = st.slider(
            "Minimum Average Score",
            min_value=float(df['average_score'].min()),
            max_value=float(df['average_score'].max()),
            value=float(df['average_score'].min()),
            help="Set a minimum threshold for average scores"
        )
        
        test_prep = st.multiselect(
            "Test Preparation",
            options=df['test_preparation_course'].unique(),
            default=df['test_preparation_course'].unique(),
            help="Filter by test preparation course completion status"
        )
        
        lunch_type = st.multiselect(
            "Lunch Type",
            options=df['lunch'].unique(),
            default=df['lunch'].unique(),
            help="Filter by student lunch program type"
        )

# Apply filters
filtered_df = df[
    (df['gender'].isin(selected_gender)) &
    (df['race_ethnicity'].isin(selected_ethnicity)) &
    (df['parental_level_of_education'].isin(selected_parent_education)) &
    (df['average_score'] >= score_threshold) &
    (df['test_preparation_course'].isin(test_prep)) &
    (df['lunch'].isin(lunch_type))
]

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Introduction",
    "📊 Overview Analysis",
    "🔍 Detailed Insights",
    "📈 Performance Trends"
])

with tab1:
    st.title("📚 Student Performance Analysis Dashboard")
    
    st.markdown("""
    ### Welcome to the Student Performance Analysis Dashboard!
    
    This interactive tool helps you explore and understand student performance across various subjects
    and demographics. Here's what you can do:
    
    1. **Filter Data**: Use the sidebar controls to focus on specific student groups
    2. **Explore Trends**: Navigate through the tabs to see different aspects of student performance
    3. **Download Results**: Export filtered data for further analysis
    
    #### Dataset Overview:
    - **Academic Scores**: Math, Reading, and Writing
    - **Demographics**: Gender, Race/Ethnicity
    - **Background**: Parental Education, Lunch Type
    - **Preparation**: Test Preparation Course Status
    """)
    
    st.info("👈 Start by selecting filters in the sidebar, then explore each tab!")

with tab2:
    st.header("📊 Performance Overview")
    st.markdown("Quick summary of key metrics based on your selected filters:")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Students", len(filtered_df))
    with col2:
        st.metric("Avg Math Score", f"{filtered_df['math_score'].mean():.1f}")
    with col3:
        st.metric("Avg Reading Score", f"{filtered_df['reading_score'].mean():.1f}")
    with col4:
        st.metric("Avg Writing Score", f"{filtered_df['writing_score'].mean():.1f}")
    
    # Score distribution
    st.subheader("Score Distributions")
    score_cols = ['math_score', 'reading_score', 'writing_score']
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Math", "Reading", "Writing"))
    
    for i, col in enumerate(score_cols, 1):
        fig.add_trace(
            go.Histogram(x=filtered_df[col], name=col.split('_')[0]),
            row=1, col=i
        )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key findings
    st.subheader("📌 Key Findings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance Highlights:**")
        top_performers = filtered_df[filtered_df['average_score'] >= filtered_df['average_score'].quantile(0.9)]
        st.markdown(f"""
        - Top 10% threshold: {filtered_df['average_score'].quantile(0.9):.1f}
        - Number of top performers: {len(top_performers)}
        - Subjects with highest scores: {max(score_cols, key=lambda x: filtered_df[x].mean()).split('_')[0]}
        """)
    
    with col2:
        st.markdown("**Demographics:**")
        st.markdown(f"""
        - Most common parent education: {filtered_df['parental_level_of_education'].mode()[0]}
        - Gender distribution: {(filtered_df['gender'].value_counts(normalize=True) * 100).round(1).to_dict()}
        - Test prep participation rate: {(filtered_df['test_preparation_course'] == 'Completed').mean()*100:.1f}%
        """)

with tab3:
    st.header("🔍 Detailed Performance Analysis")
    
    analysis_type = st.radio(
        "Choose Analysis Type",
        options=['Score Comparison', 'Demographic Analysis', 'Preparation Impact'],
        horizontal=True,
        help="Select the type of analysis you want to explore"
    )
    
    if analysis_type == 'Score Comparison':
        subjects = st.multiselect(
            "Select subjects to compare",
            options=score_cols,
            default=score_cols,
            help="Choose which subjects to include in the comparison"
        )
        
        fig = px.box(
            filtered_df.melt(value_vars=subjects),
            x='variable',
            y='value',
            color='variable',
            title='Score Distribution by Subject',
            labels={'variable': 'Subject', 'value': 'Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == 'Demographic Analysis':
        demographic_factor = st.selectbox(
            "Select demographic factor",
            options=['gender', 'race_ethnicity', 'parental_level_of_education'],
            help="Choose which demographic factor to analyze"
        )
        
        fig = px.violin(
            filtered_df,
            x=demographic_factor,
            y='average_score',
            color=demographic_factor,
            box=True,
            title=f'Score Distribution by {demographic_factor.replace("_", " ").title()}'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Preparation Impact
        fig = px.box(
            filtered_df,
            x='test_preparation_course',
            y='average_score',
            color='test_preparation_course',
            facet_col='lunch',
            title='Impact of Test Preparation and Lunch Type on Scores'
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("📈 Performance Trends and Patterns")
    
    # Correlation analysis
    st.subheader("Subject Score Correlations")
    correlation_matrix = filtered_df[score_cols].corr()
    fig = px.imshow(
        correlation_matrix,
        labels=dict(color="Correlation"),
        title="Subject Score Correlations"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance factors
    st.subheader("Key Performance Factors")
    factor_analysis = pd.DataFrame({
        'Factor': ['Test Preparation', 'Lunch Type', 'Parent Education'],
        'Impact Score': [
            filtered_df.groupby('test_preparation_course')['average_score'].mean().max() -
            filtered_df.groupby('test_preparation_course')['average_score'].mean().min(),
            filtered_df.groupby('lunch')['average_score'].mean().max() -
            filtered_df.groupby('lunch')['average_score'].mean().min(),
            filtered_df.groupby('parental_level_of_education')['average_score'].mean().max() -
            filtered_df.groupby('parental_level_of_education')['average_score'].mean().min()
        ]
    })
    
    fig = px.bar(
        factor_analysis,
        x='Factor',
        y='Impact Score',
        title='Impact of Different Factors on Student Performance',
        color='Factor'
    )
    st.plotly_chart(fig, use_container_width=True)

# Add download button for filtered data
st.sidebar.markdown("---")
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name="filtered_student_performance.csv",
    mime="text/csv",
    help="Download the currently filtered dataset as a CSV file"
)

# Summary insights based on filtered data
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Quick Insights")
st.sidebar.markdown(f"""
- **Selected Students**: {len(filtered_df)} out of {len(df)} total
- **Average Overall Score**: {filtered_df['average_score'].mean():.1f}
- **Top Subject**: {max(score_cols, key=lambda x: filtered_df[x].mean()).split('_')[0]}
""")
