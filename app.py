from time import time
from numpy import count_nonzero
import streamlit as st
import altair as alt
import pandas as pd

st.set_page_config(layout="wide")

st.header('ITER Dashboard')

@st.cache()
def get_data():
    data = pd.read_csv('data/output/counts.tsv', sep='\t', index_col=0)
    dates = pd.read_csv('data/output/dates.tsv', sep='\t', index_col=0)
    totals = pd.read_csv('data/output/totals.tsv', sep='\t', index_col=0)
    return data, dates, totals

with st.spinner('Loading data...'):
    df, dates, totals = get_data()

query_terms = df.columns
query_terms = tuple(query_terms)

st.sidebar.header('Parameters')

queries = st.sidebar.multiselect(
    'Select words',
    query_terms, default='mater'
)

if queries:
    normalize = st.sidebar.checkbox('Normalize?', value=False)
    # for k in query_sums.keys():
    #     st.markdown(f'*{k}* appears **{query_sums[k]}** times in {query_inc[k]} texts.')
    slider_min, slider_max = int(dates['dates'].min()), 770
    slider_range = st.sidebar.slider('Date Range', value=[slider_min, slider_max])

    if normalize:
        df = df.apply(lambda x: x / totals['total'] * 1000)
    else:
        pass
    df = df[df[queries] != 0][queries].dropna()

    df = df.join(dates).dropna()
    df = df[(df['dates'] >= slider_range[0]) & (df['dates'] <= slider_range[1])]
    dates = df.pop('dates')
    df.insert(0, 'date', dates)

    date_query_sums = {query: int(sum(df[query])) for query in queries}
    date_query_avgs = {query: sum(df[query])/len(df[query]) for query in queries}
    date_query_inc = {query: count_nonzero(df[query]) for query in queries}
    date_num_texts = len(df)
    
    if normalize: 
        for k in date_query_avgs.keys():
            st.markdown(f'*{k}* appears **{round(date_query_avgs[k], 4)}** times per 1000 words in {date_query_inc[k]} texts between {slider_range[0]} and {slider_range[1]}.') 
        df['date'] = df['date'].astype(int)
    else:
        for k in date_query_sums.keys():
            st.markdown(f'*{k}* appears **{date_query_sums[k]}** times in {date_query_inc[k]} texts between {slider_range[0]} and {slider_range[1]}.')
        df = df.astype(int)

    st.header('Timeline')
    if normalize:
        timeline_df = df.groupby('date').mean()
    else:
        timeline_df = df.groupby('date').sum()

    st.line_chart(timeline_df)

    st.header('Text Data')
    st.dataframe(df.sort_values(by='date'))
    