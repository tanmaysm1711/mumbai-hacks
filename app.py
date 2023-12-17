from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import requests
import os
import re
from langchain.chat_models import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
from config import OPENAI_API_TYPE, OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

llm = ""
filepath = ""


@app.route('/create_chatroom', methods=['GET'])
def create_chatroom():
    url = request.get_json().get('url')
    chatroom_id = request.get_json().get('chatroom_id')

    document_id = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url).group(1) if re.search(
        r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url) else None

    # Make the GET request to the Google Sheets CSV link
    r = requests.get(
        f'https://docs.google.com/spreadsheet/ccc?key={document_id}&output=csv')

    # Check if the request was successful (status code 200)
    if r.status_code == 200:
        # Specify the local file path where you want to save the CSV data
        local_file_path = f'./data/{chatroom_id}.csv'

        # Open the file in binary write mode and write the content to it
        with open(local_file_path, 'wb') as file:
            file.write(r.content)

        print(f'Data saved to {local_file_path}')

        return jsonify({'status': True, 'message': 'Data Saved and Chatroom created!'})
    else:
        print(f'Error: Unable to fetch data. Status code: {r.status_code}')

    return jsonify({'status': False, 'message': 'Error receiving file'})


# @app.route('/initialize_chatroom', methods=['GET'])
# def initialize_chatroom():
#     global llm
#     global filepath
#
#     # filepath = "./data/" + request.get_json().get('chatroom_id') + ".csv"
#     #
#     # llm = AzureChatOpenAI(
#     #     deployment_name="social-media-app",
#     #     model_name="gpt-35-turbo",
#     #     temperature=0,
#     # )
#     #
#     # agent = create_csv_agent(llm, filepath)
#     # session['agent'] = agent
#
#     return jsonify({'status': True, 'message': 'Chatroom initialized!'})


@app.route('/process_query', methods=['POST'])
def process_query():
    global llm
    global filepath
    query = request.get_json().get('query')
    filepath = "./data/" + request.get_json().get('chatroom_id') + ".csv"

    agent = create_csv_agent(llm, filepath)

    result = agent.run(query)

    return jsonify({'status': True, 'query_response': result})


@app.route('/analyze_chats', methods=['POST'])
def analyze_chats():
    llm2 = AzureChatOpenAI(
        deployment_name="social-media-app",
        model_name="gpt-35-turbo",
        temperature=0.7,
    )

    chats_data = request.body.chats_data

    chat_analysis_data = []

    for chat in chats_data:
        result = llm2("Chat Conversation: " + chat + "\nChat Name: [Give a Unique Name for the chat based on the things"
                                                     "discussed in the chat]\nConversion Rate: [Give a suitable value "
                                                     "between0 and 1 by analyzing the conversation which happened "
                                                     "between the user and the AI Chat Bot]\n\nJust give the Chat "
                                                     "Name and Conversion Rate. No other information is required.")

        chat_name = result['response'].split('\n')[1].split(':')[1].strip()
        conversion_rate = result['response'].split('\n')[2].split(':')[1].strip()
        chat_analysis_data.append({'chat_id': chat.id, 'chat_name': chat_name, 'conversion_rate': conversion_rate})

    return jsonify({'status': True, 'chat_analysis_data': chat_analysis_data})


if __name__ == '__main__':
    os.environ["OPENAI_API_TYPE"] = OPENAI_API_TYPE
    os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION
    os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    llm = AzureChatOpenAI(
        deployment_name="social-media-app",
        model_name="gpt-35-turbo",
        temperature=0.5,
    )

    app.run(debug=True)
