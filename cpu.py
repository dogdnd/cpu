import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
from PIL import Image

import altair as alt


row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('CPU Data EDA and ML benchmark score predict Project')

row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("made by dogdnd")
    st.markdown("used library : pandas, numpy, streamlit, matplotlib.pyplot, seaborn, joblib")
    st.markdown("[GitHub Repository](https://github.com/dogdnd/cpu)")

st.image('https://s19538.pcdn.co/wp-content/uploads/2021/09/semiconductors.jpg')


def main() :
    df = pd.read_csv('data/CPU_r23_v2.csv')

    




    st.info('컬럼별 CPU 정보 데이터프레임')
    column_list = st.multiselect('선택', df.columns)
    if len(column_list) != 0 :
        st.dataframe(df[column_list])

    st.info('제조사별 cpu 정보')
    company = df['manufacturer'].unique()
    choice = st.selectbox('선택', company)
    st.dataframe(df.loc[df['manufacturer'] == choice , ])

    
    if st.checkbox('제조사별 CPU 데이터 그래프') :

        fig = px.bar(df, x=df['cores'],title= '제조사별 코어스레드 그래프' ,y=df['threads'], color=df['manufacturer'])
        st.plotly_chart(fig)

        fig = px.bar(df, x=df['baseClock'],title= '제조사별 클럭 그래프', y=df['turboClock'], color=df['manufacturer'],)
        st.plotly_chart(fig)


        df1 = df['manufacturer'].value_counts()
        df1 = pd.DataFrame(df1)
        fig1 = px.pie(df1, names= df1.index,title='제조사별 cpu 보유 차트',values=df1['manufacturer'])
        st.plotly_chart(fig1)

    

        


    
    
    

    
    
    st.info('Clock 구간별 CPU 정보')
    baseClock_choice = st.slider('baseClock', 1.1, 4.0, 1.1 , 0.1)
    turboClock_choice = st.slider('turboClock', 3.2, 5.0 , 3.2, 0.1)
    df1 = df.loc[  df['baseClock'] <= baseClock_choice ,   ['cpuName','baseClock','turboClock','singleScore','multiScore'] ] 
    df2 = df1.loc[  df['turboClock'] <= turboClock_choice   ,   ]
    
    if st.checkbox("해당 조건에 맞는 CPU 보기") :
        st.dataframe(df2.sort_values(['baseClock','singleScore'],ascending = True))

        st.info('제조사별 Flagship CPU')
        top_cpu_by_manufacturer = df.groupby('manufacturer')[['manufacturer','singleScore','multiScore','cores','threads','baseClock','turboClock']].max()
        st.dataframe(top_cpu_by_manufacturer)



    st.info('CPU 요소별 비율 차트')
    treemap_menu = ['baseClock','turboClock','cores','threads']
    treemap_choice = st.selectbox('선택',treemap_menu)

    if st.checkbox('트리맵 차트 보기') :

    
        if treemap_choice == treemap_menu[0] :
            ax= px.treemap(df,path=["baseClock"])
            st.plotly_chart(ax)
            
        elif treemap_choice == treemap_menu[1] :
            ax= px.treemap(df,path=["turboClock"])
            st.plotly_chart(ax)

        elif treemap_choice == treemap_menu[2] :
            ax= px.treemap(df,path=["cores"])
            st.plotly_chart(ax)

        elif treemap_choice == treemap_menu[3] :
            ax= px.treemap(df,path=["threads"])
            st.plotly_chart(ax)
    
    
    
    st.info('CPU 구성 요소에 따른 상관관계 분석그래프')


    column_list = df.columns[2:-1]
    pair_choice = st.multiselect('상관관계 선택',column_list)

    if len(pair_choice) != 0 :
        st.dataframe(df[pair_choice].corr())
        
        if st.checkbox('상관관계 그래프 보기') :
            fig1 = sns.pairplot(data = df, vars = pair_choice )
            st.pyplot(fig1)


    
    st.info('ALTAIR 분포도 그래프')
    
    list2 = ['cores & threads','clocks']
    choice2 = st.selectbox('선택',list2)

    if st.checkbox('그래프 보기',) :


        if choice2 == list2[0] :
            a = alt.Chart(df).mark_circle().encode(
            x='singleScore', 
            y='multiScore', 
            size='cores', 
            color='threads', 
            tooltip=['singleScore', 'multiScore','cores','threads']
            )
            st.altair_chart(a, use_container_width=True)


        elif choice2 == list2[1] :
            a = alt.Chart(df).mark_circle().encode(
            x='singleScore', 
            y='multiScore', 
            size='baseClock', 
            color='turboClock', 
            tooltip=['singleScore', 'multiScore','baseClock','turboClock']
            )
            
            st.altair_chart(a, use_container_width=True)



    
    
    regressor = joblib.load('data/regressor.pkl')
    scaler_x = joblib.load('data/scaler_x.pkl')


    st.info('ML을 활용한 cpu 점수예측 프로그램')
    core_input = st.slider('core',max_value=64)
    thread_input = st.slider('thread',max_value=128)
    base_input = st.slider('base', 1.1, 4.0, 1.1, 0.1)
    turbo_input = st.slider('turbo', 3.2, 5.0, 3.2, 0.1)

    cpu_info = np.array([core_input,thread_input,base_input,turbo_input])
    cpu_info = cpu_info.reshape(1,4)
    cpu_pred = regressor.predict(cpu_info)
    cpu_pred= scaler_x.inverse_transform(cpu_pred)
    
    cpu_pred = cpu_pred.tolist()



    # 익스펜더
    with st.expander('벤치마크 점수 예측') :
        st.write('예측된 싱글코어 점수는' ,round(cpu_pred[0][0]), '입니다.')
        st.write('예측된 멀티코어 점수는' ,round(cpu_pred[0][1]), '입니다.')






if __name__ == '__main__' :
        main()