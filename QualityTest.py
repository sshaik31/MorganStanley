import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from io import StringIO

st.title("Morgan Stanley Quality Benchmark Tool v1.0")

uploaded_file = st.file_uploader("Choose a csv file")
if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)
    
    cd = pd.read_csv("Cost_Center.csv")
    EMP_dict = dict (zip(cd['Cost Center'], cd['Groups']))
    # EMP_dict

    df = pd.read_csv(uploaded_file)
    df1 = pd.read_csv(uploaded_file,header=5)
    df2 = df1.copy()

    df2.drop(['EMP ID', 'Cost Center','Groups'], axis = 1, errors='ignore', inplace = True)

    EMP_Split = df2['Employee ID'].str.split(pat = ';', expand = True)
    df2.insert(loc = 15, column = 'EMP ID', value = EMP_Split[0])
    df2.insert(loc = 16, column = 'Cost Center', value = EMP_Split[1])

    df2['Groups'] = df2['Cost Center'].map(EMP_dict)
    df2["Groups"].fillna("No Cost_Center", inplace = True)

    df3 = df2.copy()

    my_table = pd.pivot_table(df3, values=['Taxes in USD','Transaction Amount in USD (incl. Taxes)'],
                              index=['Service','Groups'],
                              columns='Transaction Type',
                              aggfunc='sum',fill_value=0,margins=True
                            )

    simple_view = pd.concat([
    y.append(y.sum().rename((x, 'Grand Total')))
    for x, y in my_table.groupby(level=0)
    ])                       


    unique_tripid = pd.pivot_table(df3, values='Trip/Eats ID', index='Groups', columns='Service',
               aggfunc=pd.Series.nunique,margins=True, margins_name='Grand Total',fill_value=0)


    Expanded_view = df3[(df3.Groups == 'GCM') | (df3.Groups == 'IBD1') | (df3.Groups == 'IBD2') | (df3.Groups == 'IBD3') | (df3.Groups == 'No Cost_Center')].pivot_table(index=['Service','Groups'],
                                                    values=['Fare in USD (excl. Taxes)', 'Taxes in USD','Tip in USD','Transaction Amount in USD (incl. Taxes)'],
                                                    columns=['Transaction Type'],
                                                    aggfunc='sum', margins=True, margins_name='Grand Total',fill_value=0)



    
    st.write(simple_view)
    st.write(unique_tripid)
    st.write(Expanded_view)
    
    csv = df3.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    


    st.markdown('### **⬇️ Download output CSV File **')
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
