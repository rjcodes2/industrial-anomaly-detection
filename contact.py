import streamlit as st
import pandas as pd
import os

st.title(" Contact / Help Desk")

st.markdown("""
If you have any questions, feedback, or bug reports — fill the form below.  
Your submission will be stored and reviewed. 
""")

with st.form("contact_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Email Address")
    message = st.text_area("Message or Query")
    
    submitted = st.form_submit_button("Submit")

    if submitted:
        entry = pd.DataFrame([{
            "Name": name,
            "Email": email,
            "Message": message
        }])
        
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        entry.to_csv("logs/contact_queries.csv", mode='a', header=not os.path.exists("logs/contact_queries.csv"), index=False)
        st.success(" Your message has been submitted!")
