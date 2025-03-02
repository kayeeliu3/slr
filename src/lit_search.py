import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import re
import streamlit as st
import fitz
from dotenv import load_dotenv

load_dotenv()
IEEE_API_KEY = os.getenv('IEEE_API_KEY')
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')

def search_literature(query, max_results=10):
    results = []
    
    pubmed_results = search_pubmed(query, max_results)
    results.extend(pubmed_results)

    # ieee_results = search_ieee(query, max_results)
    # results.extend(ieee_results)

    scopus_results = search_scopus(query, max_results)
    results.extend(scopus_results)
    
    arxiv_results = search_arxiv(query, max_results)
    results.extend(arxiv_results)

    df = pd.DataFrame(results)

    return df

def search_pubmed(query, max_results):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    response = requests.get(base_url, params=params).json()
    paper_ids = response.get("esearchresult", {}).get("idlist", [])

    papers = []
    for paper_id in paper_ids:
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {"db": "pubmed", "id": paper_id, "retmode": "json"}
        fetch_response = requests.get(fetch_url, params=params).json()
        
        if paper_id in fetch_response["result"]:
            data = fetch_response["result"][paper_id]
            papers.append({
                "Title": data.get("title", "N/A"),
                "Source": "PubMed",
                "DOI": data.get("elocationid", "N/A").replace("doi: ", "doi.org/"),
                "Abstract": data.get("abstract", "No Abstract Available"),
                "PDF Link": "N/A",  # PubMed does not provide direct PDFs
                "arXiv ID": "N/A"
            })
    
    return papers


def search_ieee(query, max_results):
    base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
    params = {
        "apikey": IEEE_API_KEY,
        "format": "json",
        "max_records": max_results,
        "querytext": query
    }
    response = requests.get(base_url, params=params).json()
    
    papers = []
    for article in response.get("articles", []):
        papers.append({
            "Title": article.get("title", "N/A"),
            "Source": "IEEE Xplore",
            "DOI": article.get("doi", "N/A"),
            "Abstract": article.get("abstract", "No Abstract Available"),
            "PDF Link": article.get("pdf_url", "N/A") if "pdf_url" in article else "N/A",
            "arXiv ID": "N/A"
        })
    
    return papers

def search_scopus(query, max_results):
    base_url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
    params = {"query": query, "count": max_results, "format": "json"}
    response = requests.get(base_url, headers=headers, params=params).json()
    
    papers = []
    for entry in response.get("search-results", {}).get("entry", []):
        papers.append({
            "Title": entry.get("dc:title", "N/A"),
            "Source": "Scopus",
            "DOI": "doi.org/" + entry.get("prism:doi", "N/A"),
            "Abstract": entry.get("dc:description", "No Abstract Available"),
            "PDF Link": "N/A",
            "arXiv ID": "N/A"
        })
    
    return papers

def search_arxiv(query, max_results):
    base_url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    response = requests.get(base_url, params=params).text

    root = ET.fromstring(response)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"  # Direct PDF link

        papers.append({
            "Title": title,
            "Source": "arXiv",
            "DOI": "N/A",
            "Abstract": abstract,
            "PDF Link": pdf_link,
            "arXiv ID": arxiv_id
        })
    
    return papers

def extract_conclusion_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        pdf_path = "temp_paper.pdf"
        
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()

        return extract_conclusion(text)
    
    except Exception as e:
        return "Conclusion extraction failed"

def extract_conclusion(text):
    match = re.search(r"(?i)(conclusion[s]?:?|summary and conclusion[s]?)\s*(.*?)(?=\n\n|\Z)", text, re.DOTALL)
    return match.group(2).strip() if match else "No Conclusion Available"

query = st.experimental_get_query_params().get("query", ["AI in medical diagnostics"])[0]
df = search_literature(query)

st.title("Literature Search Results")
st.dataframe(df)

