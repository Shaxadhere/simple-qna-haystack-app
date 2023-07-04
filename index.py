import requests
from docx import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.nodes import TransformersReader
from haystack.pipelines import ExtractiveQAPipeline

# Download the document from the URL
url = "http://localhost:3000/qna.docx"
response = requests.get(url)

# Save the document locally
file_path = "qna.docx"
with open(file_path, "wb") as file:
    file.write(response.content)

# Convert the document to plain text
document = Document(file_path)
text = "\n".join([paragraph.text for paragraph in document.paragraphs])

# Index the converted document with Haystack
document_store = InMemoryDocumentStore()

# Prepare the document to be indexed
doc = {"content": text}

# Index the document
document_store.write_documents([doc])

# Create the document store and index the document
document_store = InMemoryDocumentStore()
document_store.write_documents([{"content": text}])

# Create the retriever
retriever = TfidfRetriever(document_store=document_store)

# Create the reader
reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

# Create the pipeline
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Ask a question
question = "What is the name of series?"
result = pipeline.run(query=question, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 5}})

from haystack.utils import print_answers

print_answers(result, details="all", max_text_len=100)
# for answer in answers:
#     print(f"Answer: {answer['answer'][0]['answer']} - Score: {answer['answer'][0]['score']}")