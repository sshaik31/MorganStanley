import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


st.title("Cornershop SKU Classifier v2.1.1")

uploaded_file = st.file_uploader("Choose a csv file")
if uploaded_file is not None:

    training_df = pd.read_csv('Training_Data1.csv')
    test_df = pd.read_csv(uploaded_file)
    
    training_df.rename(columns={
                                'name':'item_name',
                            'parent_category_name':'entered_category',
                            'category_name': 'entered_sub_category' }
                                                                                ,inplace = True)


    training_df = training_df[["item_name", "entered_category", "entered_sub_category"]]
    test_df.rename(columns={
                                'name':'item_name' }
                                                                                ,inplace = True)
    
    test_df = test_df[["item_name"]]

    

    df_Train_null_values = training_df.loc[:,training_df.columns.isin(
                                                        ['item_name',
                                                         'entered_category',
                                                         'entered_sub_category'])]  


    null_values = df_Train_null_values[df_Train_null_values.isna().any(axis=1)]

    df_Train_null_values.dropna(inplace=True)


    duplicate = df_Train_null_values[df_Train_null_values.duplicated(
                                                                ['item_name',
                                                                 'entered_category',
                                                                 'entered_sub_category'])]


    df_Train_without_dup = df_Train_null_values.drop_duplicates(subset=['item_name',
                                                                    'entered_category',
                                                                    'entered_sub_category'])


    mismatch_data = df_Train_without_dup[df_Train_without_dup.duplicated(subset=['item_name'], keep=False)]

    group_mismatch_data = mismatch_data.groupby(
                                            ["item_name",
                                             "entered_category",
                                             "entered_sub_category"]).sum()

    Cleaned_data = df_Train_without_dup.drop_duplicates( "item_name" , keep=False)

    training_df = Cleaned_data

    test_df.dropna(inplace=True)

    def input_join_section(x):
        return (x['item_name'])

    def output_join_section(x):
        return (x['entered_category'] + ' __ ' + x['entered_sub_category'])

    training_df.loc[:,'Input'] = training_df.apply(input_join_section,axis=1)
    training_df.loc[:,'output'] = training_df.apply(output_join_section,axis=1)

    test_df.loc[:,'Input'] = test_df.apply(input_join_section,axis=1)

    

    def remove_bracket2(sample_str):
        clean4 = re.compile('\(.*?\)')
        cleantext = re.sub(clean4, ' ', sample_str)
        return cleantext.strip()


    def remove_bracket(sample_str):
        try:
            clean1 = re.compile('\(')
            cleantext = re.sub(clean1, '', sample_str)
            clean2 = re.compile('\)')
            cleantext = re.sub(clean2, '', cleantext)
        except:
            print(sample_str)
            cleantext = sample_str
        return cleantext.strip()


    def str_lower(s):
        return s.lower()
     
    def remove_comma(sample_str):
        clean4 = re.compile(',')
        cleantext = re.sub(clean4, '', sample_str)
        return cleantext.strip()


    def remove_period(sample_str):
        clean5 = re.compile('\\.')
        cleantext = re.sub(clean5, '', sample_str)
        return cleantext.strip()


    def special_characters(sample_str):
        alphanumeric = ""
        for character in sample_str:
            if character.isspace():
                alphanumeric += character
            elif character.isalnum():
                alphanumeric += character
        return alphanumeric.strip()

    def Firstletter_upper(sample_str):
        clean6 = re.compile('\/(^\s*\w|[\.\!\?]\s*\w)') #not_working
        cleantext = re.sub(clean6, ' ', sample_str)
        return cleantext.strip()    

    def titlecase(s):
        return re.sub(r"[A-Za-z]+('[A-Za-z]+)?",
        lambda word: word.group(0).capitalize(),
        s)



    def str_upper(s):
        return s.upper()

    def str_title(s):
        return s.title()
        
    def str_captalize(s):    
        return s.capitalize()

    def str_Units(s):
        clean7  = re.compile("sunfeast", re.IGNORECASE)
        cleantext = re.sub(clean7, "sunnyy", s)
        return cleantext.strip()
      

    
    training_df.loc[:,'Input'] = training_df['Input'].apply(remove_bracket2)
    training_df.loc[:,'Input'] = training_df['Input'].apply(str_lower)
    training_df.loc[:,'Input'] = training_df['Input'].apply(remove_comma)
    training_df.loc[:,'Input'] = training_df['Input'].apply(remove_period)
    training_df.loc[:,'Input'] = training_df['Input'].apply(special_characters)
    training_df.loc[:,'Input'] = training_df['Input'].str.strip()


    test_df.loc[:,'Input'] = test_df['Input'].apply(remove_bracket2)
    test_df.loc[:,'Input'] = test_df['Input'].apply(str_lower)
    test_df.loc[:,'Input'] = test_df['Input'].apply(remove_comma)
    test_df.loc[:,'Input'] = test_df['Input'].apply(remove_period)
    test_df.loc[:,'Input'] = test_df['Input'].apply(special_characters)
    

    training_df.loc[:,'output'] = training_df['output'].apply(str_lower)
    training_df.loc['output'] = training_df['output'].str.strip()

    
    Train_data = training_df
    Test_Data = test_df

    

    
    duplicate_data = training_df[training_df.duplicated(['Input',
                                                       'output'],keep = False)]

    Cleaned_data = training_df.drop_duplicates(subset=['Input',
                                                    'output'])


    mismatch_data2 = Cleaned_data[Cleaned_data.duplicated(['Input'],keep = False)] 


    mismatch_data2.groupby(["Input",
                        "output"]).sum()


    training_df = Cleaned_data.fillna('')
    training_df = training_df.drop_duplicates( "Input" , keep=False)

    input_df = training_df[['Input','output']]

    input_df['output_id'] = input_df['output'].factorize()[0]


    X_train = input_df.Input.values
    y_train = input_df.output.values
    X_test = test_df.Input.values

    X_train = pd.Series(X_train)
    y_train = pd.Series(y_train)
    X_test = pd.Series(X_test)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    Xcounts = count_vect.transform(X_test.values)

    y_pred = clf.predict(Xcounts)

    dictvarclf = pd.DataFrame(clf.predict_proba(Xcounts), columns=clf.classes_).T
    dictvarclfchanged = dictvarclf.reset_index()

    Array_main = []
    Arr_item_id = []
    Arr_proba = []
    Arr_predicted_output = []

    for (columnName, columnData) in dictvarclfchanged.iteritems():
        item_id = columnName
        max_proba = np.amax(columnData.values)
        final_proba = np.where(columnData.values == np.amax(columnData.values))[0][0]
        for (columnName, columnData) in dictvarclfchanged.iteritems():
            predicted_class = columnData.values[final_proba]
            break
        inner_array = []
        inner_array.append(item_id) 
        inner_array.append(max_proba)
        inner_array.append(predicted_class)
        Array_main.append(inner_array)
        Arr_item_id.append(item_id)
        Arr_proba.append(max_proba)
        Arr_predicted_output.append(predicted_class)

    Array_main.pop(0)
    Arr_item_id.pop(0)
    Arr_proba.pop(0)
    Arr_predicted_output.pop(0)

    column_names = ['Item_ID','Input', 'Probability', 'Predicted_Output']  

    Column_one = np.array(Arr_item_id)
    Column_two = np.array(X_test)
    Column_three = np.array(Arr_proba)
    Column_four = np.array(Arr_predicted_output)

    df5 = pd.DataFrame(np.vstack(
        [Column_one,Column_two, Column_three, Column_four]).T,
                    columns=column_names)

    table_dict = df5.to_dict(orient='records')

    table_dict = pd.DataFrame(table_dict)

    final_data = pd.merge(Test_Data, table_dict)
    Rearrange = final_data
    
    new = Rearrange["Predicted_Output"].str.split(" __ ", n = 1, expand = True)
    Rearrange["Predicted_Category"]= new[0]
    Rearrange["Predicted_Sub-Category"]= new[1]

    Rearrange.loc[:,'Modified_Item_name'] = Rearrange['item_name']

    Rearrange.loc[:,'Modified_Item_name'] = Rearrange['Modified_Item_name'].apply(str_captalize)
    #Rearrange.loc[:,'Modified_Item_name'] = Rearrange['Modified_Item_name'].apply(remove_bracket2)
    #Rearrange.loc[:,'Modified_Item_name'] = Rearrange['Modified_Item_name'].apply(titlecase)
    #Rearrange.loc[:,'Modified_Item_name'] = Rearrange['Modified_Item_name'].apply(str_title)
    #Rearrange.loc[:,'Modified_Item_name'] = Rearrange['Modified_Item_name'].apply(str_Units)
    
    Rearrange.loc[:,'Predicted_Category'] = Rearrange['Predicted_Category'].apply(titlecase)
    Rearrange.loc[:,'Predicted_Sub-Category'] = Rearrange['Predicted_Sub-Category'].apply(titlecase)
    

    df_final = Rearrange[['item_name','Modified_Item_name','Probability','Predicted_Category','Predicted_Sub-Category']]
    df_final.drop_duplicates(keep='first',inplace=True)
    
    st.write(df_final)
    
    csv = df_final.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    


    st.markdown('### **⬇️ Download output CSV File **')
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    st.markdown(href, unsafe_allow_html=True)
