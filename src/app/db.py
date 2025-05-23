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
    try:
        existing_data = collection.get(include=["metadatas"])
    except Exception as e:
        print("Error retrieving existing metadata:", e)
        existing_data = {"metadatas": []}
    
    existing_dois = set()
    for meta in existing_data.get("metadatas", []):
        doi_val = meta.get("doi", "").strip().lower()
        if doi_val and doi_val != "n/a":
            existing_dois.add(doi_val)
    
    new_ids = []
    new_metadatas = []
    new_documents = []
    
    for paper in papers:
        doi = paper.get("DOI", "").strip().lower()
        # If DOI exists and is already indexed, skip paper
        if doi and doi != "n/a" and doi in existing_dois:
            print(f"Skipping duplicate paper with DOI: {doi}")
            continue
        
        paper_id = str(uuid.uuid4())
        new_ids.append(paper_id)
        
        text = f"{paper.get('Title', '')}\n{paper.get('Abstract', '')}"
        new_documents.append(text)
        
        metadata = {
            "title": paper.get("Title", ""),
            "doi": paper.get("DOI", ""),
            "source": paper.get("Source", ""),
            "authors": paper.get("Authors", ""),
            "doi_suffix": paper.get("doi_suffix", ""),
            "PMCID": paper.get("PMCID", ""),
            "Year": paper.get("Year", "")
        }
        new_metadatas.append(metadata)
    
    if new_ids:
        embeddings = [get_embedding(doc) for doc in new_documents]
        collection.add(
            ids=new_ids,
            embeddings=embeddings,
            metadatas=new_metadatas,
            documents=new_documents
        )
        print(f"Indexed {len(new_ids)} new papers into ChromaDB.")
    else:
        print("No new papers to index.")


def query_vector_db(query_text, n_results=50):
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

