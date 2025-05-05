import streamlit as st

# Set page and Sidebar layout
st.set_page_config(layout="centered")  # Enables wider layout but not forced full width
st.html( """ <style> [data-testid="stSidebarContent"] { color: #232D4B; background-color: #E57200; } </style> """ )

# Give Dashboard a title and give user options
st.sidebar.title('UVA Career Center Data')
st.sidebar.image('uva.png', width=200, )  # Adjust the width as needed
topic = st.radio(
    'What UVA Career Center Data Do you Want to Explore? ',
    [':orange[Trends by School at UVA]', ':orange[Internships By Major]',
      ':orange[First Generation Students]', ':orange[Model]',], horizontal=True)



# Call other scripts based on user input
if topic == ':orange[Trends by School at UVA]':
    import data_by_groups
    data_by_groups.import_data()
    data_by_groups.main()

elif topic == ':orange[First Generation Students]':
    import first_gen
    first_gen.user_input()

elif topic == ':orange[Internships By Major]':
    import internships_by_major
    internships_by_major.page_choice()

elif topic == ':orange[Model]':
    import internship_impact_model
    internship_impact_model.main()

elif topic == ':orange[State Retention Data]':
    st.write(-2)

elif topic == ':orange[IPP Data]':
    exec(open('IPP_Data.py').read())
    st.write(-1)
elif topic ==':orange[Barriers]':
    exec(open('barriers.py').read())
