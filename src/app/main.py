#!/usr/bin/env python3
import json, os
import re
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, redirect, render_template, request, jsonify, url_for
from google import genai

from app import search_literature 
from db import index_papers, query_vector_db

app = Flask(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        pubmed = request.form.get("pubmed")
        pmc = request.form.get("pmc")
        scopus = request.form.get("scopus")
        springer = request.form.get("springer")
        acm = request.form.get("acm")
        search_method = request.form.get("search_method")
        
        if search_method == "vector":
            return redirect(url_for("vector_results", query=query,
                                    pubmed=pubmed, pmc=pmc, scopus=scopus,
                                    springer=springer, acm=acm))
        elif search_method == "llm":
            return redirect(url_for("llm_results", query=query,
                                    pubmed=pubmed, pmc=pmc, scopus=scopus,
                                    springer=springer, acm=acm))
            
    return render_template("index.html")

@app.route("/vector-results")
def vector_results():
    query = request.args.get("query")
    enabled_dbs = []
    if request.args.get("pubmed") is not None:
        enabled_dbs.append("pubmed")
    if request.args.get("pmc") is not None:
        enabled_dbs.append("pmc")
    if request.args.get("scopus") is not None:
        enabled_dbs.append("scopus")
    if request.args.get("springer") is not None:
        enabled_dbs.append("springer")
    if request.args.get("acm") is not None:
        enabled_dbs.append("acm")
    query_results = run_vector_query(query, query, enabled_dbs)
    return render_template("vector_results.html", results=query_results, query=query, total=len(query_results))

@app.route("/llm-results")
def llm_results():
    query = request.args.get("query")
    enabled_dbs = []
    if request.args.get("pubmed") is not None:
        enabled_dbs.append("pubmed")
    if request.args.get("pmc") is not None:
        enabled_dbs.append("pmc")
    if request.args.get("scopus") is not None:
        enabled_dbs.append("scopus")
    if request.args.get("springer") is not None:
        enabled_dbs.append("springer")
    if request.args.get("acm") is not None:
        enabled_dbs.append("acm")
        
    query_results = run_gemini_query(query, enabled_dbs)
    return render_template("llm_results.html", results=query_results, query=query, total=len(query_results))

def run_vector_query(search_query, chroma_query, enabled_dbs):
    papers_df = search_literature(search_query, max_results=25, enabled_dbs=enabled_dbs)
    papers_list = papers_df.to_dict(orient="records")  # Convert DataFrame to list of dicts

    if not papers_list:
        print("No papers found!")
        return []

    index_papers(papers_list)
    results = query_vector_db(chroma_query)

    query_results = []
    distances = results.get("distances", [[None]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    
    for i, doc in enumerate(documents):
        title = metadatas[i].get("title")
        doc_text = doc
        
        if title and doc_text.startswith(title): # Sometimes the title is included in the document text - remove this
            doc_text = doc_text[len(title):].lstrip("\n ").strip()
            
        query_results.append({
            "distance": distances[i],
            "title": title,
            "doi": metadatas[i].get("doi"),
            "source": metadatas[i].get("source"),
            "authors": metadatas[i].get("authors"),
            "document": doc_text
        })
    
    print(f"Found {len(query_results)} similar papers.")
    
    return query_results

def run_gemini_query(search_query, enabled_dbs):
    papers_df = search_literature(search_query, max_results=20, enabled_dbs=enabled_dbs)
    papers_list = papers_df.to_dict(orient="records")
    
    if not papers_list:
        print("No papers found!")
        return []
    
    print("Querying Gemini...")
    
    gemini_response = gemini_filter(search_query, papers_list)
    print("Gemini response:")
    print(gemini_response.replace("```json", "").replace("```", ""))

    try:
        similar_items = json.loads(gemini_response.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print("Error parsing Gemini output after extraction:", e)
        similar_items = []
    
    final_results = []
    for item in similar_items:
        idx = item.get("paper_index")
        explanation = item.get("explanation")
        if idx is None:
            continue
        # Convert from 1-based index to 0-based.
        i = int(idx) - 1
        if i < 0 or i >= len(papers_list):
            continue
        paper = papers_list[i]
        # Map keys to lowercase for template consistency.
        paper["title"] = paper.get("Title")
        paper["doi"] = paper.get("DOI")
        paper["abstract"] = paper.get("Abstract")
        paper["source"] = paper.get("Source")
        paper["authors"] = paper.get("Authors")
        paper["explanation"] = explanation
        final_results.append(paper)
    
    print(f"Found {len(final_results)} similar papers based on Gemini filtering.")
    return final_results

def query_gemini(query):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=query
    )
    print(response.text)

def gemini_filter(search_query, papers):
    aggregated_text = ""
    for i, paper in enumerate(papers, start=1):
        aggregated_text += (
            f"Paper {i}:\n"
            f"Title: {paper.get('Title')}\n"
            f"Abstract: {paper.get('Abstract')}\n\n"
        )
    
    prompt = (
        f"I have the following research papers (with their titles and abstracts):\n\n"
        f"{aggregated_text}\n"
        f"My search query is: \"{search_query}\".\n\n"
        f"Please return a JSON array where each item is an object with three keys:\n"
        f"  - 'paper_index': the index (starting from 1) of a paper that is similar to the search query.\n"
        f"  - 'title': title of the paper.\n\n"
        f"  - 'explanation': a brief explanation of why that paper is similar.\n\n"
        f"For example:\n"
        f"""[
  {{
    "paper_index": 2,
    "title": "Deep Learning for Medical Image Segmentation",
    "explanation": "This paper discusses deep learning for medical image segmentation."
  }},
  {{
    "paper_index": 5,
    "title": "Deep Learning in Radiology",
    "explanation": "This paper focuses on deep learning applied to radiology."
  }}
]"""
    )
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=prompt
    )
    
    return response.text

if __name__ == '__main__':
    app.run(debug=True)
