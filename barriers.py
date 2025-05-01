import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

barriers_data = pd.read_csv('Barriers_anonymous.csv')

barriers_data['First Generation Student'] = barriers_data['Q4']
barriers_data['School Year'] = barriers_data['Q3']
barriers_data['Internship Status'] = barriers_data['Q6']
barriers_data['Undergrad Research'] = barriers_data['Q2_2']
barriers_data['Clinical Experience'] = barriers_data['Q2_3']
barriers_data['Internship Location'] = barriers_data['Q11']
barriers_data['Reason To Not Pursue Internship'] = barriers_data['Q20']
#barriers_data['intership_barriers'] = barriers_data['Q13']

demographic = st.sidebar.selectbox("Which Groups do you want to filter by", ['First Generation Student', 'School Year'])
barriers_col = st.sidebar.selectbox("Which Aspect of Barriers Survey do you want to compare", 
                                ['Internship Status', 'Undergrad Research', 'Clinical Experience',
                                 'Internship Location', 'Reason To Not Pursue Internship'])
method = st.sidebar.selectbox("How do you want to see this data", ['Counts', 'Percentages'])


#st.write("Table 1: " + barriers_col + ' among ' + demographic + 's')
st.write("Table 1:", barriers_data[barriers_col][0], divider=True)
if demographic == 'First Generation Student':
    if method == 'Counts':
        ct = pd.crosstab(barriers_data[demographic], barriers_data[barriers_col]).reset_index()
        ct = ct.fillna(' ')
        ct = ct.iloc[1:-1, :-2]
        st.write(ct.T)
    elif method == 'Percentages':
        ct = pd.crosstab(barriers_data[demographic], barriers_data[barriers_col])
        ct = ct.iloc[1:-1, :-2] # Get rid of junk in data
        ct.iloc[0, :] = ct.iloc[0, :]/2734 # Other students Row
        ct.iloc[1, :] = ct.iloc[1, :]/653 # Row First Gen
        ct = ct.applymap(lambda x: f"{x * 100:.2f}%") # Multiply and limit decimals
        ct = round(ct.reset_index(), 2)
        st.write(ct.T)

else:
    if method == 'Counts':
        ct = pd.crosstab(barriers_data[demographic], barriers_data[barriers_col])
        ct = ct.loc[['Fourth Year', 'Second Year', 'Third Year']].reset_index()
        st.write(ct.iloc[:, :-1].T)
    elif method == 'Percentages':
        ct = pd.crosstab(barriers_data[demographic], barriers_data[barriers_col])
        ct = ct.loc[['Fourth Year', 'Second Year', 'Third Year']]# Get rid of junk in data
        ct.loc['Fourth Year'] = ct.loc['Fourth Year'] /873
        ct.loc['Third Year'] = ct.loc['Third Year']/1301
        ct.loc['Second Year'] = ct.loc['Second Year']/1218
        ct = ct.applymap(lambda x: f"{x * 100:.2f}%") # Multiply and limit decimals
        ct = round(ct.reset_index(), 2)
        st.write(ct.iloc[:, :-1.T)

