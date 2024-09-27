import streamlit as st
import pandas as pd
import numpy as np
import time

st.title('Title')
st.header('Header')
st.subheader('subheader')

st.write('Write Somethong')

text_input = st.text_input('Text를 입력해주세요.')
st.write(text_input)

password_input = st.text_input('암호를 입력해주세요', type='password')

number_input = st.number_input('숫자를 입력해주세요')
st.write(number_input)

st.date_input('날짜를 입력홰주세요.')
st.time_input('시간을 입력해주세요.')

uploaded_file = st.file_uploader(
    'Choose a file',
    type=['png', 'jpg', 'jpeg'],
)

selected_item = st.radio(
    'Radio part',
    ('A', 'B', 'C')
)

if selected_item == 'A':
    st.write('A!')
elif selected_item == 'B':
    st.write('B!')
elif selected_item == 'C':
    st.write('C!')

option = st.selectbox(
    'Please select in selectedbox!',
    ('kyle', 'roun', 'andante'),
)

st.write('You selected:', option)

multi_select = st.multiselect(
    'multi',
    ['A', 'B', 'C', 'D']
)

st.write('You selected:', multi_select)

values = st.slider(
    'Select a range of values',
    0.0,
    100.0,
    (25.0, 75.0)
)

st.write('Values:', values)

checkbox_btn = st.checkbox('checkbox btn', value=True)

if checkbox_btn:
    st.write('click!')


if st.button('if clicked'):
    st.write('message')

if st.button('if clicked 2'):
    st.write('message 2')

# form
with st.form(key='입력 form'):
    username = st.text_input('user name')
    password = st.text_input('password', type='password')
    st.form_submit_button('login')

df = pd.DataFrame({
    'first col': [1, 2, 3, 4],
    'second col': [10, 20, 30, 40]
})

st.markdown('==========')

st.write(df)
st.dataframe(df.style.highlight_max(axis=0))
st.table(df)

st.metric('My metric', 42, 2)

st.json(df.to_json())

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)

st.map(map_data)

st.caption('this is caption')
st.code('a= 123')
st.latex('\int a x^w \,dx')

with st.spinner('please wait...'):
    time.sleep(5)

st.balloons()

# status box
st.success('success')
st.info('info')
st.warning('warning')
st.error('error message')

st.sidebar.button('hi')

# session state -> global variable 처럼 공유할 수 있는 변수 만들고 저장
# 상태 유지, 특정 변수 공유, 로그인 상태 유지
# 대화 히스토리 유지, 여러 단계 form

# 안쓰는 경우
# st.title('counter example')

# count_value = 0

# increment = st.button('increment')
# if increment:
#     count_value += 1

# decrement = st.button('decrement')
# if decrement:
#     count_value -= 1

# st.write('count:', count_value)

# 쓰는 경우
st.title('counter example with session state')

if 'count' not in st.session_state:
    st.session_state.count = 0

increment = st.button('increment')
if increment:
    st.session_state.count += 1

decrement = st.button('decrement')
if decrement:
    st.session_state.count -= 1

st.write('count=', st.session_state.count)

# ui가 변경될 시 모든 코드를 다시 실행하는 특성 때문에 생기는 이슈
# 캐싱 데코레이터 활용

# 함수 호출 결과를 local에 저장해 앱이 더 빨라짐
# 캐싱된 값은 모든 사용자가 사용
# 사용자마다 다르게 접근해야 하면 session state에 저장

# @st.cahce_data
# @st.cache_resource

