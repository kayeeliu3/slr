#!/usr/bin/env python3
import json, os, re
import pandas as pd
from flask import Flask, redirect, render_template, request, jsonify, url_for
from dotenv import load_dotenv
from google import genai
import requests
from unidecode import unidecode

from app import search_literature 
from db import index_papers, query_vector_db

app = Flask(__name__)
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SPRINGER_API_KEY = os.getenv('SPRINGER_API_KEY')
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')

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

@app.route("/extraction")
def extraction_results():
    db = request.args.get("db")
    extraction_data = {}
    
    if db == "pubmed":
        pmcid = request.args.get("pmcid")
        if not pmcid:
            return "PMCID not provided.", 400
        extraction_data = extract_data_from_full_text("pubmed", pmcid=pmcid)
    elif db == "springer":
        doi = request.args.get("doi")
        if not doi:
            return "DOI not provided.", 400
        extraction_data = extract_data_from_full_text("springer", doi=doi)
    elif db == "scopus":
        doi = request.args.get("doi")
        if not doi:
            return "DOI not provided.", 400
        extraction_data = extract_data_from_full_text("scopus", doi=doi)
    elif db == "europe":
        europe_pmc = request.args.get("pmcid")
        if not europe_pmc:
            return "Europe PMC ID not provided.", 400
        extraction_data = extract_data_from_full_text("europe", pmcid=europe_pmc)
    else:
        return "Unsupported database.", 400

    return render_template("extraction.html", extraction=extraction_data)


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
        
        if title and doc_text.startswith(title):  # Sometimes the title is included in the document text - remove this
            doc_text = doc_text[len(title):].lstrip("\n ").strip()
            
        query_results.append({
            "distance": distances[i],
            "title": title,
            "doi": metadatas[i].get("doi"),
            "source": metadatas[i].get("source"),
            "authors": metadatas[i].get("authors"),
            "document": doc_text,
            "doi_suffix": metadatas[i].get("doi_suffix"),
            "PMCID": metadatas[i].get("PMCID")
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
        model="gemini-2.0-flash",
        contents=query
    )
    return unidecode(response.text)

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

def fetch_full_text(type, pmcid=None, doi=None):
    if type == "pubmed":
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error fetching full text for PMC ID {pmcid}: {response.status_code} - {response.text}")
    elif type == "springer":
        base_url = "https://api.springernature.com/openaccess/jats"
        params = {
            "api_key": SPRINGER_API_KEY,
            "q": f"doi:{doi}"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error fetching Springer full text for DOI {doi}: {response.status_code} - {response.text}")
    elif type == "scopus":
        base_url = "https://api.elsevier.com/content/article/doi/"
        url = f"{base_url}{doi}"
        headers = {
            "X-ELS-APIKey": SCOPUS_API_KEY,
            "Accept": "application/xml"  # Change to "application/json" if JSON is preferred.
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error fetching Scopus full text for DOI {doi}: {response.status_code} - {response.text}")
    elif type == "europe":
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            response = requests.get(url)
            if response.status_code == 200:
                return response.text
            else:
                raise Exception(f"Error fetching full text for Europe PMC ID {pmcid}: {response.status_code} - {response.text}")


# def chunk_text(text, chunk_size=1000):
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
def gemini_extract_data(chunk):
    prompt = (
        f"Extract only the statistics, key data points, and a summary of the whole paper from the following text, "
        f"strictly based on the information provided. Do not add any additional details not mentioned from the paper. "
        f"Return your answer in valid JSON format with the keys 'statistics' and 'key_points' and 'summary'.\n"
        f"For each statistic, please ONLY have a brief sentence of what the statistic is about based on the relevant text (assume the reader has not read the full text, so please also provide context for this statistic).\n"
        f"The 'summary' key should be a brief summary of the whole paper.\n"
        f"If the provided document indicates the paper is not available, then return N/A as the summary.\n"
        f"For example:\n"
        f"""{{
        "statistics": [
            {{
            "statistic": "The H&E DL Angio model achieves a Spearman correlation of 0.77 with the Angioscore across multiple cohorts.",
            "context": "This statistic describes the correlation between the deep learning model's predictions and the RNA-based Angioscore in real-world data."
            }},
            {{
            "statistic": "The H&E DL Angio model achieves a Spearman correlation of 0.73 with the Angioscore across multiple cohorts.",
            "context": "This statistic describes the correlation between the deep learning model's predictions and the RNA-based Angioscore in the IMmotion150 trial cohort."
            }},
            {{
            "statistic": "The c-index for DL Angio prediction of anti-angiogenic therapy response approximates the Angioscore (0.66 vs 0.67) in the IMmotion150 trial.",
            "context": "This data point compares the performance of the deep learning model and the RNA Angioscore in predicting response to anti-angiogenic therapy, showing they are similar."
            }},
            {{
            "statistic": "Spearman correlation between CD31 IHC and the Angioscore is 0.62 in the IMmotion150 cohort.",
            "context": "The article notes that immunohistochemical staining of endothelial cells using CD31 correlates with the Angioscore with this value, though the relationship was weaker than expected"
            }},
            {{
            "statistic": "Precision, Recall, and F1 scores for CD31 arm are 0.53, 0.66, and 0.58 respectively.",
            "context": "This statistic describes the model's performance in CD31 segmentation, showing a tendency to overpredict the boundaries of the CD31 mask."
            }},
            {{
            "statistic": "The H&E DL Angioscore has a correlation of 0.68 with the RNA Angioscore on the held-out portion of the training TCGA cohort.",
            "context": "This metric illustrates the correlation achieved by the model on a subset of the TCGA dataset that was not used for training."
            }},
            {{
            "statistic": "Overall survival of 520 patients is stratified by H&E DL Angioscore, with a c-index of 0.75",
            "context": "This data shows how well the H&E DL Angioscore can predict overall survival"
            }}
        ],
        "key_points": [
            "The study developed a deep learning (DL) model (H&E DL Angio) to predict anti-angiogenic (AA) therapy response in renal cancer using histopathology slides.",
            "H&E DL Angio achieves strong correlation with the RNA-based Angioscore, a predictor of treatment response.",
            "The model predicts AA response in clinical trial cohorts, performing similarly to the Angioscore but at a lower cost.",
            "Angiogenesis inversely correlates with grade and stage in renal cancer.",
            "The model provides a visual representation of the vascular network, enhancing interpretability."
        ],
        "summary": "This paper presents a deep learning model (H&E DL Angio) that predicts anti-angiogenic therapy response in renal cancer patients based on histopathology slides. The model correlates strongly with the RNA-based Angioscore, predicts treatment response in clinical trials, and reveals biological insights such as the inverse relationship between angiogenesis and tumor grade/stage. It offers a cost-effective and interpretable alternative to RNA-based assays for predicting therapy response."
        }}\n"""
        "Text:\n" + chunk
    )
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    print(f"RESPONSE: {response.text}")
    try:
        extraction = json.loads(response.text.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print("Error parsing Gemini extraction response:", e)
        extraction = {"statistics": "", "key_points": ""}
    return extraction


def extract_data_from_full_text(type, doi=None, pmcid=None):
    if type == "pubmed":
        full_text = fetch_full_text("pubmed", pmcid=pmcid)
        print(f"Full text acquired for PMC ID: {pmcid}.")
    if type == "springer":
        full_text = fetch_full_text("springer", doi=doi)
        print(f"Full text acquired for Springer with DOI: {doi}.")
    if type == "scopus":
        full_text = fetch_full_text("scopus", doi=doi)
        print(f"Full text acquired for Scopus with DOI: {doi}.")
    if type == "europe":
        full_text = fetch_full_text("europe", pmcid=pmcid)
        print(f"Full text acquired for Europe PMC ID: {pmcid}.")
    if type == "acm":
        pass
    extracted_results = []
    
    extraction = gemini_extract_data(full_text)
    extracted_results.append(extraction)
    
    aggregated_statistics = []
    aggregated_key_points = []
    summary = []
    
    for res in extracted_results:
        stat = res.get("statistics", {})
        if isinstance(stat, list):
            aggregated_statistics.extend(stat)  # flatten the list
        else:
            aggregated_statistics.append(stat)
        kp = res.get("key_points", [])
        if isinstance(kp, list):
            aggregated_key_points.extend(kp)
        else:
            aggregated_key_points.append(kp)
    
    return {"statistics": aggregated_statistics, "key_points": aggregated_key_points, "summary": res.get("summary")}

if __name__ == '__main__':
    app.run(debug=True)
