import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import hashlib 
import chardet
import numpy as np


USER_DATA = {"testuser": hashlib.sha256("testpassword".encode()).hexdigest()} 


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def detect_encoding(file):
    raw_data = file.read(10000)  
    result = chardet.detect(raw_data)
    file.seek(0)  
    return result['encoding']


def advanced_clean_data(df, missing_thresh=0.5, fill_num_option='mean', fill_cat_option='unknown', drop_outliers=False):
    st.write("Initial Data Overview:")
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())


    df_cleaned = df.drop_duplicates()
    st.write(f"Removed duplicates. New shape: {df_cleaned.shape}")

   
    missing_percent = df_cleaned.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > missing_thresh].index
    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
    st.write(f"Dropped columns with more than {missing_thresh*100}% missing values. Remaining columns: {df_cleaned.columns.tolist()}")

   
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    if fill_num_option in ['mean', 'median', 'mode']:
        if fill_num_option == 'mean':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
        elif fill_num_option == 'median':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
        elif fill_num_option == 'mode':
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mode().iloc[0])

    df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna(fill_cat_option)
    st.write("Filled missing values based on the selected options.")

   
    if drop_outliers:
        for col in numeric_cols:
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        st.write(f"Removed outliers based on IQR. New shape: {df_cleaned.shape}")
    else:
        st.write("Outliers were not removed.")

    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].str.lower()

    if 'date' in df_cleaned.columns:
        df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')

    st.write("Final Data Overview:")
    buffer = StringIO()
    df_cleaned.info(buf=buffer)
    st.text(buffer.getvalue())

    return df_cleaned


def download_csv(data):
    csv = data.to_csv(index=False)
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    return b


def generate_dashboard(df):
    st.header('Advanced Dashboard')


    st.write(f"Total rows in dataset: {len(df)}")

   
    st.sidebar.header('Chart Axis Configuration')
    common_x_axis = st.sidebar.selectbox('Select column for x-axis:', df.columns.tolist())
    common_y_axis = st.sidebar.selectbox('Select column for y-axis:', df.columns.tolist())

    st.subheader('Interactive Bar Chart')
    chart_color = st.sidebar.color_picker('Pick a color for bar chart', '#1f77b4')
    if common_x_axis and common_y_axis:
        fig = px.bar(df, x=common_x_axis, y=common_y_axis, title=f"Bar Chart: {common_x_axis} vs {common_y_axis}", color_discrete_sequence=[chart_color])
        st.plotly_chart(fig, use_container_width=True)

   
    st.subheader('Interactive Scatter Plot')
    scatter_color = st.selectbox('Select column for color coding (optional):', [None] + df.columns.tolist())
    if common_x_axis and common_y_axis:
        fig = px.scatter(df, x=common_x_axis, y=common_y_axis, color=scatter_color, title=f"Scatter Plot: {common_x_axis} vs {common_y_axis}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Dynamic Line Chart')
    if common_x_axis and common_y_axis:
        fig = px.line(df, x=common_x_axis, y=common_y_axis, title=f"Line Chart: Trend over {common_x_axis}")
        st.plotly_chart(fig, use_container_width=True)

   
    st.subheader('Advanced Pie Chart')
    pie_values = st.selectbox('Select column for pie chart values:', df.columns.tolist(), key='pie_values')
    pie_names = st.selectbox('Select column for pie chart names (optional):', [None] + df.columns.tolist(), key='pie_names')
    if pie_values:
        fig = px.pie(df, values=pie_values, names=pie_names, title=f"Pie Chart: Distribution of {pie_values}")
        st.plotly_chart(fig, use_container_width=True)

    
    st.subheader('Donut Chart')
    donut_values = st.selectbox('Select column for donut chart values:', df.columns.tolist(), key='donut_values')
    donut_names = st.selectbox('Select column for donut chart names (optional):', [None] + df.columns.tolist(), key='donut_names')
    if donut_values:
        fig = px.pie(df, values=donut_values, names=donut_names, hole=0.3, title=f"Donut Chart: Distribution of {donut_values}")
        st.plotly_chart(fig, use_container_width=True)

  
    st.subheader('Advanced Box Plot')
    if common_x_axis and common_y_axis:
        fig = px.box(df, x=common_x_axis, y=common_y_axis, title=f"Box Plot: {common_x_axis} vs {common_y_axis}")
        st.plotly_chart(fig, use_container_width=True)

  
    st.subheader('Histogram')
    hist_col = st.selectbox('Select column for histogram:', df.columns.tolist(), key='hist_col')
    if hist_col:
        fig = px.histogram(df, x=hist_col, title=f"Histogram of {hist_col}")
        st.plotly_chart(fig, use_container_width=True)

  
    st.subheader('Tree Map')
    tree_map_path = st.multiselect('Select columns for tree map path (hierarchical):', df.columns.tolist(), default=df.columns.tolist()[:2])
    tree_map_values = st.selectbox('Select values for the tree map:', df.columns.tolist(), key='tree_map_values')
    if tree_map_path and tree_map_values:
        fig = px.treemap(df, path=tree_map_path, values=tree_map_values, title=f"Tree Map: {tree_map_values}")
        st.plotly_chart(fig, use_container_width=True)

    
    st.subheader('Correlation Heatmap')
    corr_matrix = df.corr()
    if len(corr_matrix) > 0:
        
        if len(corr_matrix) > 100:
            corr_matrix = corr_matrix.sample(n=100, axis=1).sample(n=100, axis=0)
        fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)

   
   
    st.sidebar.header('Download Data')
    st.download_button(label='Download Filtered CSV', 
                       data=download_csv(df), 
                       file_name='filtered_data.csv', 
                       mime='text/csv')


def login_page():
    st.title("Login / Sign Up")
    option = st.selectbox("Select an option", ["Login", "Sign Up"])

    if option == "Sign Up":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            if username and password:
                if username in USER_DATA:
                    st.error("Username already exists. Please choose another one.")
                else:
                    USER_DATA[username] = hash_password(password)
                    st.success("User registered successfully! You can now log in.")
            else:
                st.error("Please fill in all fields.")
    
    elif option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            hashed_input_password = hash_password(password)
            if username in USER_DATA and USER_DATA[username] == hashed_input_password:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
                st.experimental_rerun() 
            else:
                st.error("Invalid credentials.")
    
    
    st.sidebar.header("Simulate Google Sign-In")
    if st.sidebar.button("Simulate Google Sign-In"):
        st.session_state['logged_in'] = True
        st.experimental_rerun()  


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    st.title('Advanced File Cleaner and Dashboard Generator')

   
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['csv', 'xls', 'xlsx'])

   
    st.sidebar.title('Cleaning Options')
    missing_thresh = st.sidebar.slider("Missing value threshold (%)", 0, 100, 50) / 100
    fill_num_option = st.sidebar.selectbox("Filling option for numeric columns", ['mean', 'median', 'mode'])
    fill_cat_option = st.sidebar.selectbox("Filling option for categorical columns", ['unknown', 'mode', 'none'])
    drop_outliers = st.sidebar.checkbox("Remove outliers", value=False)

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = file_name.split('.')[-1].lower()
        
        encoding = detect_encoding(uploaded_file)
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file, encoding=encoding)
        else:
            df = pd.read_excel(uploaded_file)

      
        st.write(f"Cleaning data from file: {file_name}")
        df_cleaned = advanced_clean_data(df, missing_thresh, fill_num_option, fill_cat_option, drop_outliers)

      
        st.subheader("Cleaned Data")
        st.dataframe(df_cleaned)

        
        generate_dashboard(df_cleaned)

        st.download_button(label='Download Cleaned CSV',
                           data=download_csv(df_cleaned),
                           file_name='cleaned_data.csv',
                           mime='text/csv')

else:
    login_page()

