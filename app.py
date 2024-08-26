import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 제목 설정
st.title('데이터 시각화 앱')

# 데이터셋 선택
dataset = st.selectbox(
    '분석할 데이터셋을 선택하세요',
    ('iris', 'titanic', 'diamonds')
)

# 선택된 데이터셋 로드
@st.cache_data
def load_data(dataset):
    if dataset == 'iris':
        return sns.load_dataset('iris')
    elif dataset == 'titanic':
        return sns.load_dataset('titanic')
    else:
        return sns.load_dataset('diamonds')

data = load_data(dataset)

# 데이터프레임 표시
st.subheader(f'{dataset} 데이터셋')
st.write(data.head())

# 기본 통계 정보
st.subheader('기본 통계 정보')
st.write(data.describe())

# 열 선택
columns = st.multiselect('그래프로 표시할 열을 선택하세요', data.columns)

# 선택된 열에 대한 히스토그램
if columns:
    st.subheader('히스토그램')
    fig, ax = plt.subplots()
    for col in columns:
        sns.histplot(data[col], kde=True, ax=ax)
    st.pyplot(fig)

    # 선택된 열 간의 산점도
    if len(columns) >= 2:
        st.subheader('산점도')
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=columns[0], y=columns[1], ax=ax)
        st.pyplot(fig)

# 상관관계 히트맵
if st.checkbox('상관관계 히트맵 보기'):
    st.subheader('상관관계 히트맵')
    
    # 숫자형 열만 선택
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    # 범주형 데이터 인코딩 (예: sex 열)
    if 'sex' in data.columns:
        data['sex_encoded'] = data['sex'].map({'male': 0, 'female': 1})
        numeric_columns = numeric_columns.append(pd.Index(['sex_encoded']))
    
    # 선택된 숫자형 열에 대해서만 상관관계 계산
    correlation = data[numeric_columns].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # 상관관계 해석 추가
    st.subheader('상관관계 해석')
    st.write("""
    - 1에 가까울수록 강한 양의 상관관계
    - -1에 가까울수록 강한 음의 상관관계
    - 0에 가까울수록 상관관계가 약함
    """)

    # 가장 강한 상관관계 표시
    st.subheader('가장 강한 상관관계')
    correlation_sorted = correlation.unstack().sort_values(key=abs, ascending=False)
    correlation_sorted = correlation_sorted[correlation_sorted != 1.0]  # 자기 자신과의 상관관계 제외
    st.write(correlation_sorted.head(5))