# Imports
from numpy import count_nonzero
import streamlit as st
# import altair as alt
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly
import plotly.graph_objs as go

top_n = 20

# @st.cache 
def append_list(sim_words, words):
    
    list_of_words = []
    
    for i in range(len(sim_words)):
        
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)
        
    return list_of_words

def display_scatterplot_2D(model, user_input=None, words=None, label=None, color_map=None, annotation='On', dim_red = 'PCA', perplexity = 0, learning_rate = 0, iteration = 0, topn=0, sample=10):

    word_vectors = np.array([model[w] for w in words])

    if dim_red == 'PCA':
        two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
    else:
        two_dim = TSNE(random_state=0, perplexity = perplexity, learning_rate = learning_rate, n_iter = iteration).fit_transform(word_vectors)[:,:2]

    data = []
    count = 0
    for i in range (len(user_input)):

                trace = go.Scatter(
                    x = two_dim[count:count+topn,0], 
                    y = two_dim[count:count+topn,1],  
                    text = words[count:count+topn] if annotation == 'On' else '',
                    name = user_input[i],
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 15,
                        'opacity': 0.8,
                        'color': 2
                    }
       
                )
               
                data.append(trace)
                count = count+topn

    trace_input = go.Scatter(
                    x = two_dim[count:,0], 
                    y = two_dim[count:,1],  
                    text = words[count:],
                    name = 'input words',
                    textposition = "top center",
                    textfont_size = 20,
                    mode = 'markers+text',
                    marker = {
                        'size': 25,
                        'opacity': 1,
                        'color': 'black'
                    }
                    )

    data.append(trace_input)
    
# Configure the layout.
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white", 
            font_size=20, 
            font_family="Courier New"),
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1600,
        height = 800,
        )


    plot_figure = go.Figure(data = data, layout = layout)

    st.plotly_chart(plot_figure)




 

# Streamlit app
st.set_page_config(layout="wide")

st.header('ITER Dashboard')

# @st.cache()
def get_models():
    modelfile = 'data/models/allLASLAlemmi-vector-100-nocase-w10-SKIP.bin'
    model=KeyedVectors.load_word2vec_format(modelfile, binary=True)
    return model

@st.cache()
def get_data():
    data = pd.read_csv('data/output/counts.tsv', sep='\t', index_col=0)
    dates = pd.read_csv('data/output/dates.tsv', sep='\t', index_col=0)
    modes = pd.read_csv('data/output/modes.tsv', sep='\t', index_col=0)
    totals = pd.read_csv('data/output/totals.tsv', sep='\t', index_col=0)
    return data, dates, modes, totals

with st.spinner('Loading data...'):
    df, dates, modes, totals = get_data()
    model = get_models()

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
            empty_df = True
            df = df_state
            slider_min, slider_max = slider_min_state, slider_max_state
            valid_slider = False
        else:
            empty_df = False

        dates = df.pop('date')

        df.insert(0, 'date', dates)
        df.insert(1, 'mode', mode_col)

        date_query_sums = {query: int(sum(df[query])) for query in queries}

        if not empty_df:
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

            st.header('Semantic neighbors')

            result_word = []
            user_input = queries.copy()

            missing_words = [f'*{item}*' for item in user_input if item not in model]
            user_input = [item for item in user_input if item in model]
            

            print(missing_words)

            if missing_words:
                st.markdown(f'NB: {", ".join(missing_words)} do not appear in model vocabulary.')

            for words in user_input:
                sim_words = model.most_similar(words, topn = top_n)
                # sim_words = [sim_word for sim_word in sim_words if sim_word[0] not in queries]
                
                sim_words = append_list(sim_words, words)
                result_word.extend(sim_words)

                similar_word = [word[0] for word in result_word]
                similarity = [word[1] for word in result_word] 
                similar_word.extend(user_input)
                labels = [word[2] for word in result_word]
                label_dict = dict([(y,x+1) for x,y in enumerate(set(labels))])
                color_map = [label_dict[x] for x in labels]   

            # dim_red = st.sidebar.selectbox('Select dimension reduction method', ('PCA','TSNE')) # TSNE not working correctly
            dim_red = st.sidebar.selectbox('Select dimension reduction method', ['PCA'])

            display_scatterplot_2D(model, queries, similar_word,labels, color_map, annotation="On", dim_red=dim_red, perplexity=30, learning_rate=200, iteration=1000, topn=top_n) 

            def horizontal_bar(word, similarity):
                
                similarity = [ round(elem, 2) for elem in similarity ]
                
                data = go.Bar(
                        x= similarity,
                        y= word,
                        orientation='h',
                        text = similarity,
                        marker_color= 4,
                        textposition='auto')

                layout = go.Layout(
                        font = dict(size=20),
                        xaxis = dict(showticklabels=False, automargin=True),
                        yaxis = dict(showticklabels=True, automargin=True,autorange="reversed"),
                        margin = dict(t=20, b= 20, r=10)
                        )

                plot_figure = go.Figure(data = data, layout = layout)
                st.plotly_chart(plot_figure)            

            st.header('The Top 5 Most Similar Words for Each Input')
            count=0
            for i in range (len(user_input)):
                
                st.write('The most similar words from '+str(user_input[i])+' are:')
                horizontal_bar(similar_word[count:count+5], similarity[count:count+5])
                
                count = count+top_n

    else:
        'Please select at least one mode from the sidebar.'
else:
    st.markdown('Please select at least one query term from the sidebar.')
    