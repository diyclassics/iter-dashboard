# Imports
from numpy import count_nonzero
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

# Streamlit app
st.set_page_config(layout="wide")

st.header('ITER Dashboard')

@st.cache()
def get_data():
    data = pd.read_csv('data/output/counts.tsv', sep='\t', index_col=0)
    dates = pd.read_csv('data/output/dates.tsv', sep='\t', index_col=0)
    modes = pd.read_csv('data/output/modes.tsv', sep='\t', index_col=0)
    totals = pd.read_csv('data/output/totals.tsv', sep='\t', index_col=0)
    return data, dates, modes, totals

with st.spinner('Loading data...'):
    df, dates, modes, totals = get_data()

query_terms = df.columns
query_terms = tuple(query_terms)

st.sidebar.header('Parameters')

queries = st.sidebar.multiselect(
    'Select words',
    query_terms, default='mater'
)

if queries:
    modes_ = st.sidebar.multiselect('Select mode', options=('prose','verse'), default=('prose','verse'))

    if queries and modes_:
        normalize = st.sidebar.checkbox('Normalize?', value=False)    
        combine_mode = st.sidebar.checkbox('Combine mode?', value=False)
        slider_min, slider_max = int(dates['date'].min()), 770
        slider_range = st.sidebar.slider('Date Range', value=[slider_min, slider_max])
        valid_slider = True
        slider_min, slider_max = slider_range

        if normalize:
            df = df.apply(lambda x: x / totals['total'] * 1000)

        df = df[queries]
        df = df.loc[~(df==0).all(axis=1)]

        df = df.join(modes).dropna()
        df = df[df['mode'].isin(modes_)]
        mode_col = df.pop('mode')

        df = df.join(dates).dropna()
        df_state = df.copy()
        slider_min_state, slider_max_state = slider_min, slider_max
        df = df[(df['date'] >= slider_min) & (df['date'] <= slider_max)]
        if len(df) < 1:
            st.warning('There are no entries for these queries in this date range. Select a different range')
            df = df_state
            slider_min, slider_max = slider_min_state, slider_max_state
            valid_slider = False
        dates = df.pop('date')

        df.insert(0, 'date', dates)
        df.insert(1, 'mode', mode_col)

        date_query_sums = {query: int(sum(df[query])) for query in queries}
        date_query_avgs = {query: sum(df[query])/len(df[query]) for query in queries}        
        date_query_inc = {query: count_nonzero(df[query]) for query in queries}
        date_num_texts = len(df)
        
        if valid_slider:
            if normalize: 
                for k in date_query_avgs.keys():
                    st.markdown(f'*{k}* appears **{round(date_query_avgs[k], 4)}** times per 1000 words in {date_query_inc[k]} texts between {slider_min} and {slider_max}.') 
                
            else:
                for k in date_query_sums.keys():
                    st.markdown(f'*{k}* appears **{date_query_sums[k]}** times in {date_query_inc[k]} texts between {slider_min} and {slider_max}.')

            df['date'] = df['date'].astype(int)

            if not normalize:
                df[queries] = df[queries].astype(int)

            st.header('Timeline')
            
            if combine_mode:
                df = pd.pivot_table(df, 
                    values=queries,
                    index = ['date'],
                    columns = ['mode'],
                    aggfunc = np.sum)        
                df.columns = ['_'.join(col) for col in df.columns]
                df = df.fillna(0)
                if not normalize:
                    df = df.astype(int)
                st.line_chart(df)
                # st.dataframe(df)
            else:
                if normalize:
                    timeline_df = df.groupby('date').mean()
                else:
                    timeline_df = df.groupby('date').sum()

                st.line_chart(timeline_df)

            st.header('Text Data')
            st.dataframe(df.sort_values(by='date'))
    else:
        'Please select at least one mode from the sidebar.'
else:
    st.markdown('Please select at least one query term from the sidebar.')
    