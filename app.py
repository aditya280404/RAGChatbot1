import streamlit as st
import os
import pandas as pd
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import bert_score
from datasets import load_metric, load_dataset
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("AthinaAI Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only. 
Please provide the most accurate response based on the question.

<context> {context} <context>
Questions: {input}
""")


def create_embeddings():
    if "vectors" not in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        loader = PyPDFLoader("policy-booklet-0923.pdf")
        docs = loader.load()
        
        num_pages = len(docs)
        st.write(f"Loaded {num_pages} pages")
        
        
        if num_pages == 0:
            st.error("No pages loaded. Please check the PDF file.")
            return
        
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        final_documents = text_splitter.split_documents(docs)
        
        num_chunks = len(final_documents)
        

        if num_chunks == 0:
            st.error("No text chunks created. Please check the text splitting process.")
            return

       
        for i, chunk in enumerate(final_documents):
            if not chunk.page_content.strip():
                st.error(f"Chunk {i+1} is empty. There might be an issue with the text splitting process.")
                return
        
        
        vectors = FAISS.from_documents(final_documents, embeddings)
        
        if not vectors:
            st.error("Embeddings were not created. Please check the embedding creation process.")
            return
        
        st.session_state.vectors = vectors
        st.write("Vector Store DB Updated")
    else:
        st.write("Vector Store DB already exists.")
        

if st.button(" Find Embeddings first"):
    create_embeddings()


prompt1 = st.text_input("Enter Your Prompt")


if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response["answer"])
    else:
        st.write("Please find embeddings first by clicking the 'Find Embeddings' button.")

if st.button("Evaluate"):
    # Read dataset
    df = pd.read_csv("dataset.csv")
    df['question'] = df['question'].astype(str)
    df['ground_truth'] = df['ground_truth'].astype(str)

    
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        answers = []
        contexts = []

       
        for i in range(df.shape[0]):
            question = df.iloc[i, 0]
            ground_truth = df.iloc[i, 1]
            if isinstance(question, float):
                question = str(question)
            if isinstance(ground_truth, float):
                ground_truth = str(ground_truth)
            response = retrieval_chain.invoke({'input': question})
            answers.append(response['answer'])
            contexts.append([doc.page_content for doc in response["context"]])

        df['answer'] = answers
        df['contexts'] = contexts
        df['ground_truth'] = df.iloc[:, 1]

      
        P, R, F1 = bert_score.score(df['answer'].tolist(), df['ground_truth'].tolist(), lang='en', rescale_with_baseline=True)
        df['Precision'] = P.tolist()
        df['Recall'] = R.tolist()
        df['F1 Score'] = F1.tolist()

        
        rouge = load_metric('rouge')
        bleu = load_metric('bleu')
        rouge_scores = []
        bleu_scores = []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for idx, (answer, ground_truth) in enumerate(zip(df['answer'], df['ground_truth'])):
            rouge_score = scorer.score(answer, ground_truth)
            rouge_scores.append(rouge_score)

            answer_tokens = [answer.split()]
            ground_truth_tokens = [ground_truth.split()]
            bleu_score = bleu.compute(predictions=answer_tokens, references=[ground_truth_tokens])
            bleu_scores.append(bleu_score)

        df['ROUGE-1 Precision'] = [score['rouge1'].precision for score in rouge_scores]
        df['ROUGE-1 Recall'] = [score['rouge1'].recall for score in rouge_scores]
        df['ROUGE-1 F1'] = [score['rouge1'].fmeasure for score in rouge_scores]
        df['ROUGE-2 Precision'] = [score['rouge2'].precision for score in rouge_scores]
        df['ROUGE-2 Recall'] = [score['rouge2'].recall for score in rouge_scores]
        df['ROUGE-2 F1'] = [score['rouge2'].fmeasure for score in rouge_scores]
        df['ROUGE-L Precision'] = [score['rougeL'].precision for score in rouge_scores]
        df['ROUGE-L Recall'] = [score['rougeL'].recall for score in rouge_scores]
        df['ROUGE-L F1'] = [score['rougeL'].fmeasure for score in rouge_scores]
        df['BLEU'] = [score['bleu'] for score in bleu_scores]

        
        if 'ground_truth' in df.columns:
            df.to_csv("dataset_with_answers.csv", index=False)
            st.write("Dataset processed and saved with answers and evaluation metrics.")
            st.write("Evaluation Results:")
            st.write(df[['question', 'answer', 'ground_truth', 'Precision', 'Recall', 'F1 Score', 
                         'ROUGE-1 Precision', 'ROUGE-1 Recall', 'ROUGE-1 F1', 
                         'ROUGE-2 Precision', 'ROUGE-2 Recall', 'ROUGE-2 F1', 
                         'ROUGE-L Precision', 'ROUGE-L Recall', 'ROUGE-L F1', 'BLEU']])
        else:
            st.write("The dataset must contain a 'ground_truth' column.")
    else:
        st.write("Please find embeddings first by clicking the 'Find Embeddings' button.")
        
if st.button("Fine tune"):
    
    df = pd.read_csv("dataset.csv")
    df['question'] = df['question'].astype(str)
    df['ground_truth'] = df['ground_truth'].astype(str)

    # Process dataset if vectors exist
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        answers = []
        contexts = []

       
        for i in range(df.shape[0]):
            question = df.iloc[i, 0]
            ground_truth = df.iloc[i, 1]
            if isinstance(question, float):
                question = str(question)
            if isinstance(ground_truth, float):
                ground_truth = str(ground_truth)
            response = retrieval_chain.invoke({'input': question})
            answers.append(response['answer'])
            contexts.append([doc.page_content for doc in response["context"]])

        df['answer'] = answers
        df['contexts'] = contexts
        df['ground_truth'] = df.iloc[:, 1]

       
        model_name = "t5-small"  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        def preprocess_function(examples):
            inputs = [str(ex) for ex in examples['question']]
            targets = [str(ex) for ex in examples['ground_truth']]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        
        train_data_path = 'train_data.csv'
        eval_data_path = 'eval_data.csv'
        df[['question', 'ground_truth']].sample(frac=0.8, random_state=42).to_csv(train_data_path, index=False)
        df[['question', 'ground_truth']].drop(df[['question', 'ground_truth']].sample(frac=0.8, random_state=42).index).to_csv(eval_data_path, index=False)

        
        train_dataset = load_dataset('csv', data_files={'train': train_data_path}).map(preprocess_function, batched=True)
        eval_dataset = load_dataset('csv', data_files={'train': eval_data_path}).map(preprocess_function, batched=True)

        
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir="./logs",
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=eval_dataset["train"],
            data_collator=data_collator
        )

        trainer.train()

       
        model.save_pretrained("fine_tuned_model")

        
        eval_results = trainer.evaluate()
        st.write("Fine-tuning completed and model saved as 'fine_tuned_model'.")
        st.write("Evaluation Results:")
        st.write(eval_results)
    else:
        st.write("Please find embeddings first by clicking the 'Find Embeddings' button.")
