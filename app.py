# import openai
import streamlit as st
from streamlit_chat import message
import requests
import json
# from streamlit_star_rating import st_star_rating
from trubrics.integrations.streamlit import FeedbackCollector


DATABRICKS_SP_TOKEN = st.secrets["DATABRICKS_SP_TOKEN"]
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
MODEL_NAME = st.secrets["MODEL_NAME"]

def create_tf_serving_json(data):
  return {'inputs': {'question': [data]}}

def ask_question(dataset):
  url = f'{DATABRICKS_HOST}/serving-endpoints/{MODEL_NAME}/invocations'
  headers = {'Authorization': f'Bearer {DATABRICKS_SP_TOKEN}', 'Content-Type': 'application/json'}

  data_json = json.dumps(create_tf_serving_json(dataset), allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# Setting page title and header
st.set_page_config(page_title="LLM-Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>LLM based chatbot</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4 (Coming soon)"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = ask_question(prompt)

    response = completion["predictions"][0]["answer"]
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion["predictions"][0]["output_metadata"]["token_usage"]["total_tokens"]
    prompt_tokens = completion["predictions"][0]["output_metadata"]["token_usage"]["prompt_tokens"]
    completion_tokens = completion["predictions"][0]["output_metadata"]["token_usage"]["completion_tokens"]
    return response, total_tokens, prompt_tokens, completion_tokens


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

collector = FeedbackCollector(
            component_name="LLM Feedback Demo",
            email=st.secrets["TRUBRICS_EMAIL"],  # Store your Trubrics credentials in st.secrets:
            password=st.secrets["TRUBRICS_PASSWORD"],  # https://blog.streamlit.io/secrets-in-sharing-apps/
        )

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Ask')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
            # stars = st_star_rating(label="Please rate you experience", maxValue=5, key=f"rating-{i}", defaultValue=0)
            # st.write(stars)

            collector.st_feedback(
                feedback_type="thumbs",
                model=f"{MODEL_NAME}",
                open_feedback_label="[Optional] Provide additional feedback",
                key = f"feedback-{i}" # each key should have a unique id
            )
