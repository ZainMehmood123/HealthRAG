import streamlit as st
import os
import json
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

# Define paths
dataset_path = r'E:\Desktop\asssss\Assignment 5\mimic-iv-ext-direct-1.0.0'
samples_path = os.path.join(dataset_path, 'samples', 'Finished')
diagnostic_kg_path = os.path.join(dataset_path, 'diagnostic_kg', 'Diagnosis_flowchart')

# Preprocessing function
def extract_relevant_data(data, is_diagnostic_kg=True):
    if is_diagnostic_kg:
        diagnostic_info = data.get('diagnostic', {})
        knowledge_info = data.get('knowledge', {})
        risk_factors = "No risk factors available"
        symptoms = "No symptoms available"
        for key, value in knowledge_info.items():
            if isinstance(value, dict):
                if 'Risk Factors' in value:
                    risk_factors = value['Risk Factors']
                if 'Symptoms' in value:
                    symptoms = value['Symptoms']
                if risk_factors == "No risk factors available" and 'Risk' in value:
                    risk_factors = value['Risk']
                if symptoms == "No symptoms available" and 'Signs' in value:
                    symptoms = value['Signs']
        diagnostic_details = ""
        if diagnostic_info:
            for key, value in diagnostic_info.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        diagnostic_details += f"{sub_key}: {sub_value if sub_value else 'No diagnostic details available'}\n"
                else:
                    diagnostic_details += f"{key}: {value if value else 'No diagnostic details available'}\n"
        return risk_factors, symptoms, diagnostic_details
    else:
        note_text = data.get('text', 'No clinical note available')
        return "N/A", "N/A", note_text

# Load and preprocess data
documents = []
document_metadata = []
for file_name in os.listdir(diagnostic_kg_path):
    if file_name.endswith('.json'):
        with open(os.path.join(diagnostic_kg_path, file_name), 'r') as file:
            data = json.load(file)
            risk_factors, symptoms, diagnostic_details = extract_relevant_data(data, is_diagnostic_kg=True)
            document = f"Risk Factors: {risk_factors}\nSymptoms: {symptoms}\nDiagnostic Details: {diagnostic_details}"
            tokens = word_tokenize(document.lower())
            documents.append(tokens)
            document_metadata.append({
                "source": "diagnostic_kg",
                "disease": file_name.replace(".json", ""),
                "text": document
            })
for disease_folder in os.listdir(samples_path):
    disease_path = os.path.join(samples_path, disease_folder)
    if os.path.isdir(disease_path):
        for pdd_folder in os.listdir(disease_path):
            pdd_path = os.path.join(disease_path, pdd_folder)
            if os.path.isdir(pdd_path):
                for file_name in os.listdir(pdd_path):
                    if file_name.endswith('.json'):
                        with open(os.path.join(pdd_path, file_name), 'r') as file:
                            data = json.load(file)
                            risk_factors, symptoms, note_text = extract_relevant_data(data, is_diagnostic_kg=False)
                            document = f"Clinical Note: {note_text}"
                            tokens = word_tokenize(document.lower())
                            documents.append(tokens)
                            document_metadata.append({
                                "source": "samples",
                                "disease": disease_folder,
                                "pdd": pdd_folder,
                                "text": document
                            })

# Initialize BM25
bm25 = BM25Okapi(documents)

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Retrieval function
def retrieve_documents(query, top_k=5):
    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved_docs = [document_metadata[idx] for idx in top_indices]
    return retrieved_docs, scores[top_indices]

# Generation helper functions
def extract_field_from_text(text, field):
    """Extract a specific field (e.g., Symptoms, Risk Factors) from the document text and simplify it."""
    lines = text.split("\n")
    for line in lines:
        if line.startswith(f"{field}:"):
            content = line[len(f"{field}:"):].strip()
            content = content.split(";")[0].strip()
            content = content.replace("Typical: ", "").replace("Less typical: ", "")
            content = content.replace(", ", ", ").replace(". ", ", ")
            return content
    return "Not available"

def determine_relevant_field(query):
    """Determine which field (Symptoms, Risk Factors, Diagnostic Details) is relevant for the query."""
    query_lower = query.lower()
    if "symptom" in query_lower:
        return "Symptoms"
    elif "risk factor" in query_lower:
        return "Risk Factors"
    elif "diagnos" in query_lower:
        return "Diagnostic Details"
    else:
        return "Symptoms"

def is_answer_relevant(answer, query):
    """Check if the generated answer is relevant to the query."""
    query_lower = query.lower()
    answer_lower = answer.lower()
    if "heart failure" in query_lower and "heart" not in answer_lower:
        return False
    if "pneumonia" in query_lower and "pneumonia" not in answer_lower:
        return False
    if "symptom" in query_lower and "symptom" not in answer_lower and "sign" not in answer_lower:
        return False
    if "risk factor" in query_lower and "risk" not in answer_lower:
        return False
    if "diagnos" in query_lower and "diagnos" not in answer_lower:
        return False
    return True

def generate_answer(query, retrieved_docs, max_length=250):
    relevant_field = determine_relevant_field(query)
    fallback_answer = "Not available"
    if retrieved_docs:
        top_doc = retrieved_docs[0]
        fallback_answer = extract_field_from_text(top_doc['text'], relevant_field)
    context = f"Query: {query}\n\nRelevant Information:\n"
    for doc in retrieved_docs[:3]:
        disease = doc.get('disease', 'Unknown Disease')
        field_content = extract_field_from_text(doc['text'], relevant_field)
        context += f"Disease: {disease} - {relevant_field}: {field_content}\n"
    if "symptom" in query.lower():
        context += f"\nList the symptoms of the disease mentioned in the query using the information above:\n"
    elif "risk factor" in query.lower():
        context += f"\nList the risk factors for the disease mentioned in the query using the information above:\n"
    elif "diagnos" in query.lower():
        context += f"\nDescribe how the disease mentioned in the query is diagnosed using the information above:\n"
    else:
        context += f"\nAnswer the query using the information above:\n"
    inputs = tokenizer(
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
        return_attention_mask=True
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = answer.rfind(":\n") + 2
    if answer_start != -1:
        answer = answer[answer_start:].strip()
    else:
        answer = answer[len(context):].strip()
    if not is_answer_relevant(answer, query):
        st.warning("Generated answer was irrelevant. Falling back to retrieved information.")
        answer = fallback_answer
    return answer

# Streamlit UI
st.title("🧠 HealthRAG – Clinical Diagnostic Reasoning Assistant")
st.write("Ask a clinical question to retrieve relevant documents and generate a diagnostic response.")

query = st.text_input("🔍 Enter your query:", "What are the symptoms of heart failure?")
if st.button("Submit"):
    with st.spinner("Processing your query..."):
        retrieved_docs, scores = retrieve_documents(query)
        
        st.subheader("📄 Retrieved Documents")
        for doc, score in zip(retrieved_docs, scores):
            st.write(f"**Score:** {score:.2f}, **Source:** {doc['source']}, **Disease:** {doc.get('disease', 'N/A')}")
            st.write(f"**Text:** {doc['text']}")
            st.write("---")
        
        answer = generate_answer(query, retrieved_docs)
        st.subheader("🧾 Generated Answer")
        st.write(answer)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "🤖 <strong>HealthRAG</strong> – A Clinical Diagnostic Reasoning System <br>"
    "🔗 <a href='https://github.com/your-github-username/your-repo' target='_blank'>View on GitHub</a> 🚀"
    "</div>",
    unsafe_allow_html=True
)
