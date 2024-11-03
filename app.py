import intellikit as ik
import pandas as pd
import streamlit as st
import time

df = pd.read_json("hf://datasets/MakTek/Customer_support_faqs_dataset/train_expanded.json", lines=True)

#Define you similarity calculation methods for your project
cosine = ik.sim_sentence_cosine

# Assign the similarity calculation function to the feature
similarity_functions = {
    'question': cosine
}

# Weighting out feature. Since this is just one feature the default weight of one should be used.
feature_weights = {
    'question': 1
}


# Streamlit app
st.title("FAQs Bot")

# User input
user_question = st.text_input("Ask a question:")

if user_question:
    # Define the query
    query = pd.DataFrame({
        'question': [user_question]
    })

    # Retrieve the answer
    top_n = 1
    returned_df = ik.linearRetriever(df, query, similarity_functions, feature_weights, top_n)
    returned_dict = ik.dataframe_to_dict(df=returned_df, orientation="records")
    response = returned_dict[0]['answer']

    # Display the response with typing animation
    placeholder = st.empty()
    displayed_text = ""
    for char in response:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(0.05)  # Adjust the delay as needed

    