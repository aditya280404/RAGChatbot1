## AthinaAiChatbot

AthinaAiChatbot is a Streamlit application designed to provide conversational responses based on provided prompts and a document's context. Follow the steps below to set up and run the application:

 Setup Instructions:

1.Create a Virtual Environment:
   - Create a new virtual environment using your preferred tool (e.g., virtualenv, conda).

2. Install Dependencies:
   - Use the command `pip install -r requirements.txt` to install all the necessary dependencies specified in the `requirements.txt` file.

3. Obtain API Keys:
   - Obtain API keys from GROQ and Google Makersuite.
   
4. Configure API Keys:
   - Create a `.env` file in the project directory.
   - Add your GROQ API key and Google API key to the `.env` file in the following format:
     ```
     GROQ_API_KEY="<your_GROQ_API_key>"
     GOOGLE_API_KEY="<your_Google_API_key>"
     ```

5. Run the Application:
   - Execute the command `streamlit run app.py` to start the Streamlit application.

6. Ensure File Placement:
   - Make sure that all necessary files, including `app.py`, `requirements.txt`, and any other relevant files, are present in the same directory.

Usage Instructions:

1. Find Embeddings:
   - Click the "Find Embeddings" button in the application interface to initiate the process of creating embeddings from the provided PDF document.

2. Provide Prompts:
   - After embeddings are successfully generated, input your prompt in the designated text input field.
   - The application will use the provided prompt and document context to generate a response.

3. Evaluation:
   - Click the "Evaluate" button to evaluate the performance of the chatbot based on a provided dataset.
   - The application will process the dataset, compare the chatbot responses to ground truth, and display evaluation metrics such as precision, recall, F1 score, ROUGE scores, and BLEU score.

4. Fine-tuning (Optional):
   - Click the "Fine tune" button to fine-tune a pre-trained model using a provided dataset.
   - The fine-tuned model will be saved, and evaluation results will be displayed.

##Notes:

- Ensure that the PDF document specified in the code (`policy-booklet-0923.pdf`) is present in the directory or update the filename/path accordingly.
- The application utilizes Streamlit for the user interface and integrates various libraries such as `langchain`, `FAISS`, `GoogleGenerativeAIEmbeddings`, and `transformers` for natural language processing tasks.
- This README provides basic setup and usage instructions. Further customization and advanced usage may require familiarity with the underlying libraries and codebase.
