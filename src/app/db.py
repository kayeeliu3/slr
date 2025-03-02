import os
import json
import chromadb
from chromadb.config import Settings
from flask import jsonify
import uuid
from sentence_transformers import SentenceTransformer
from app import search_literature  

client = chromadb.Client()
collection = client.get_or_create_collection(name="papers") 
model = SentenceTransformer('all-MiniLM-L6-v2') # Model to encode text into vectors
print(f"ChromaDB collection: {collection}")

def get_embedding(text):
    return model.encode(text).tolist()

def index_papers(papers):
    ids = []
    metadatas = []
    documents = []

    for i, paper in enumerate(papers):
        paper_id = str(uuid.uuid4())
        ids.append(paper_id)
        
        text = f"{paper.get('Title', '')}\n{paper.get('Abstract', '')}"
        documents.append(text)
        
        metadata = {
            "title": paper.get("Title", ""),
            "doi": paper.get("DOI", ""),
            "source": paper.get("Source", ""),
            "authors": paper.get("Authors", "")
        }
        metadatas.append(metadata)
    
    embeddings = [get_embedding(doc) for doc in documents]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    print(f"Indexed {len(ids)} papers into ChromaDB.")

def query_vector_db(query_text, n_results=100):
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    return results

def test():
    search_query = "deep learning in medical imaging"
    papers_df = search_literature(search_query, max_results=15)
    
    papers_json = papers_df.to_dict(orient="records")
    
    if papers_df.empty:
        print("No papers found!")
        exit(0)
    
    index_papers(papers_json)
    
    sample_query = "AI for radiology and image analysis"
    similar_papers = query_vector_db(sample_query, n_results=5)
    
    print("Query results from ChromaDB:")
    print(json.dumps(similar_papers, indent=2))

