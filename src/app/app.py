from typing import Iterable
from flask import session
import os, requests, re, time
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import unicodedata, json, ast
from torrequest import TorRequest
from google import genai

load_dotenv()
IEEE_API_KEY = os.getenv('IEEE_API_KEY')
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')
SPRINGER_API_KEY = os.getenv('SPRINGER_API_KEY')
GEMINI_API_SECRET = os.getenv('GEMINI_API_SECRET')

def refine_db_queries(user_query, additional_keywords=""):
    if additional_keywords and additional_keywords.strip():
        combined_query = f"{user_query} AND {additional_keywords}"
    else:
        combined_query = user_query

    prompt = (
        f"Given the following search query: '{combined_query}', generate refined keyword search queries for the following academic literature databases (do not include publication date), ensuring each query adheres to the database's unique search syntax and standards:\n"
        "- PubMed: Use appropriate MeSH terms and boolean operators for medical literature.\n"
        "- Europe PMC: Optimise the query with filters such as IN_PMC and HAS_DOI, ensuring it suits Europe PMC's search interface.\n"
        "- Scopus: Provide a refined query string using Scopus's advanced search syntax.\n"
        "- Springer: Tailor a refined query string for the Springer API, emphasising relevant academic literature. Ensure the search is allowed within the free tier of Springer. Eg: keyword:'deep learning' AND keyword:'medical imaging' OR keyword:'diagnostic imaging' OR keyword:'radiography'\n\n"
        "Return strictly only a JSON object with keys 'pubmed', 'europe', 'scopus', and 'springer', where each key maps to its corresponding refined query string. Output only valid JSON."
    )
    client = genai.Client(api_key=GEMINI_API_SECRET)
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    try:
        print(response.text)
        refined_queries = json.loads(response.text.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print("Error parsing JSON from Gemini response:", e)
        # Fallback: use the combined query if parsing fails
        refined_queries = {
            "pubmed": combined_query,
            "europe": combined_query,
            "scopus": combined_query,
            "springer": combined_query
        }
    return refined_queries

def search_literature(query, max_results=40, enabled_dbs=None):
    if enabled_dbs is None:
        enabled_dbs = ["pubmed", "europe", "scopus", "springer"]
    results = []

    refined_queries = session.get("refined_queries")
    if not refined_queries:
        refined_queries = refine_db_queries(query)
        session["refined_queries"] = refined_queries
        
    filtered_refined = {db: refined_queries[db] for db in enabled_dbs if db in refined_queries}
    session["refined_queries"] = filtered_refined 
    
    with TorRequest(proxy_port=9050, ctrl_port=9051, password=None) as tr:
        print("Tor session established. IP:", tr.get('http://ipecho.net/plain').text)
        
        if "pubmed" in enabled_dbs:
            refined_pubmed = refined_queries.get("pubmed", query)
            print("Searching PubMed with query:", refined_pubmed)
            results.extend(search_pubmed(refined_pubmed, max_results))
            print("PubMed search complete.")
        if "europe" in enabled_dbs:
            refined_europe = refined_queries.get("europe", query)
            print("Searching Europe PMC with query:", refined_europe)
            results.extend(search_europe_pmc(refined_europe, max_results))
            print("Europe PMC search complete.")
        if "scopus" in enabled_dbs:
            refined_scopus = refined_queries.get("scopus", query)
            print("Searching Scopus with query:", refined_scopus)
            results.extend(search_scopus(tr, refined_scopus, max_results))
            print("Scopus search complete.")
        if "springer" in enabled_dbs:
            refined_springer = refined_queries.get("springer", query)
            print("Searching Springer with query:", refined_springer)
            results.extend(search_springer(refined_springer, max_results))
            print("Springer search complete.")
        
        print("All searches complete. Removing duplicates and combining output...")
        df = pd.DataFrame(results)
        df = remove_duplicates(df)
        return df

def normalise_text(text):
    # If text is a dictionary, join its string values.
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values() if isinstance(v, (str, int, float)))
    # If it's not already a string, convert it.
    elif not isinstance(text, str):
        text = str(text)
    # Now perform the unicode normalization.
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().strip()

def remove_duplicates(df):
    seen_dois = set()
    seen_composites = set()
    unique_rows = []

    for idx, row in df.iterrows():
        # Normalizse the DOI. If it's "N/A" or empty, treat it as missing.
        doi = row.get("DOI", "").strip().lower()
        doi = doi if doi and doi != "n/a" else None

        # Normalise title and authors for composite matching.
        title = row.get("Title", "").strip().lower()
        authors = row.get("Authors", "").strip().lower()
        composite_key = f"{title}||{authors}"

        # If a valid DOI has already been seen, skip this row.
        if doi and doi in seen_dois:
            print(f"Skipping duplicate DOI: {doi}")
            continue

        # If the composite key (title and authors) has already been seen, skip.
        if composite_key in seen_composites:
            print(f"Skipping duplicate composite: {composite_key}")
            continue

        # Otherwise, mark this entry as seen.
        seen_composites.add(composite_key)
        if doi:
            seen_dois.add(doi)

        unique_rows.append(row)

    return pd.DataFrame(unique_rows)

# def refine_scopus_query(user_query):
#     """
#     Use Gemini to generate a refined Scopus search query based on the user's input.
#     The prompt instructs Gemini to output a concise query string with specific keywords.
#     """
#     prompt = (
#         f"Given the following search query: '{user_query}', generate a refined, targeted keyword search string "
#         "for Scopus that focuses on relevant academic literature. Ensure it is properly URL encoded, and only return the refined query string, nothing else."
#     )
#     client = genai.Client(api_key=GEMINI_API_SECRET)
#     response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
#     refined_query = response.text.strip()
#     return refined_query if refined_query else user_query


def search_pubmed(query, max_results):
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    esearch_response = requests.get(esearch_url, params=params).json()
    paper_ids = esearch_response.get("esearchresult", {}).get("idlist", [])
    
    if not paper_ids:
        return []
    
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(paper_ids),
        "retmode": "xml"
    }
    efetch_response = requests.get(efetch_url, params=params)
    root = ET.fromstring(efetch_response.content)
    
    papers = []
    for article in root.findall("PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_info = medline.find("Article")
        
        # Retrieve PMCID from PubmedData (if available)
        pmcid = "N/A"
        pubmed_data = article.find("PubmedData")
        if pubmed_data is not None:
            article_id_list = pubmed_data.find("ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("ArticleId"):
                    if article_id.get("IdType") == "pmc":
                        pmcid_raw = article_id.text.strip()
                        # Remove the "PMC" prefix if present
                        if pmcid_raw.upper().startswith("PMC"):
                            pmcid = pmcid_raw[3:]
                        else:
                            pmcid = pmcid_raw
                        break
        
        # Title extraction
        title_elem = article_info.find("ArticleTitle")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "N/A"
        
        # Abstract extraction; if no abstract is found, skip this paper.
        abstract_elem = article_info.find("Abstract/AbstractText")
        if abstract_elem is None or not abstract_elem.text or not abstract_elem.text.strip():
            continue
        abstract = abstract_elem.text.strip()
        
        journal_issue = article_info.find("Journal/JournalIssue")
        pub_date = journal_issue.find("PubDate") if journal_issue is not None else None
        year = "N/A"
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            medline_date_elem = pub_date.find("MedlineDate")

            if year_elem is not None and year_elem.text:
                year = year_elem.text.strip()
            elif medline_date_elem is not None and medline_date_elem.text:
                year_match = re.match(r"(\d{4})", medline_date_elem.text.strip())
                if year_match:
                    year = year_match.group(1)
        
        
        # Authors extraction
        authors = []
        for author in article_info.findall("AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None and fore is not None:
                authors.append(f"{fore.text} {last.text}")
            elif author.find("CollectiveName") is not None:
                authors.append(author.find("CollectiveName").text)
        authors_str = ", ".join(authors) if authors else "No Authors Available"
        
        # DOI extraction
        doi = "N/A"
        for el in article_info.findall("ELocationID"):
            if el.get("EIdType") == "doi":
                doi = f"https://doi.org/{el.text}"
                break
        
        papers.append({
            "Title": normalise_text(title),
            "Source": "PubMed",
            "PMCID": normalise_text(pmcid),
            "DOI": normalise_text(doi),
            "Abstract": normalise_text(abstract),
            "Authors": normalise_text(authors_str),
            "Year": year
        })
    
    print(f"Found {len(papers)} papers from PubMed.")
    return papers


def search_europe_pmc(query, max_results):
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    search_term = f"{query} AND IN_PMC:y AND HAS_DOI:y"
    params = {
        "query": search_term,
        "format": "json",
        "pageSize": max_results,
        "resultType": "core"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        data = response.json()
    except Exception as e:
        print(f"Error retrieving or parsing Europe PMC response: {e}")
        return []
    
    results = []
    records = data.get("resultList", {}).get("result", [])
    for record in records:
        doi = record.get("doi", "N/A")
        if doi == "N/A" or not doi:
            continue
        
        europe_pmc_id = record.get("pmcid", 'N/A')
        title = record.get("title", "N/A")
        source = record.get("source", "N/A") + " (Europe PMC)"
        
        abstract = record.get("abstractText", "")
        if not abstract.strip():
            continue
        abstract = BeautifulSoup(abstract, "html.parser").get_text(separator=" ", strip=True)
        
        authors = "No Authors Available"
        author_list = record.get("authorList", {}).get("author", [])
        if author_list:
            author_names = [author.get("fullName", "").strip() for author in author_list if author.get("fullName", "").strip()]
            if author_names:
                authors = ", ".join(author_names)
                
        year = record.get("pubYear")
        
        paper = {
            "Title": normalise_text(title),
            "Source": normalise_text(source), 
            "PMCID": europe_pmc_id,
            "DOI": f"https://doi.org/{doi}",
            "Abstract": normalise_text(abstract),
            "Authors": normalise_text(authors),
            "Year": year
        }
        results.append(paper)
    
    print(f"Found {len(results)} papers from Europe PMC.")
    return results

def get_scopus_text(doi):
    url=f"https://api.elsevier.com/content/article/doi/{doi}"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
    
    response = requests.get(url, headers=headers)
    
    return response

def search_scopus(tr, query, max_results):
    base_url = "https://api.elsevier.com/content/search/scopus"
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY}
    params = {"query": query.replace('```', "").strip(), "count": max_results, "format": "json"}
    
    response = requests.get(base_url, headers=headers, params=params).json()

    papers = []
    for entry in response.get("search-results", {}).get("entry", []):
        print(entry)
        doi_suffix = entry.get("prism:doi", "")
        doi_url = f"https://doi.org/{doi_suffix}" if doi_suffix else "N/A"
        abstract = entry.get("dc:description", "")
        title = entry.get("dc:title", "N/A")
        
        full_text = get_scopus_text(doi_suffix)
        if full_text.status_code == "200":
            abstract = full_text.get("full-text-retrieval-response").get("coredata").get("dc:description", "")
            print("AFTER FULL:" + abstract)
            
        if not abstract and doi_url != "N/A":
            abstract = scrape_abstract_from_doi(doi_url, tr, title=title)
            
        if not abstract or not abstract.strip():
            continue
        
        year  = entry.get("prism:coverDate", "")
        
        raw_authors = entry.get("dc:creator", "No Authors Available")
        if isinstance(raw_authors, list):
            # Join the list into a comma-separated string.
            authors_str = ", ".join(raw_authors)
        else:
            authors_str = raw_authors
        print("RAW Authors:", authors_str)
        print("NORMALISE:" + normalise_text(authors_str))
        papers.append({
            "Title": normalise_text(title),
            "Source": "Scopus",
            "DOI": doi_url,
            "doi_suffix": doi_suffix,
            "Abstract": normalise_text(abstract.strip().replace("Abstract","",1)),
            "Authors": normalise_text(entry.get("dc:creator", "No Authors Available")),
            "Year": year
        })
    
    print(f"Found {len(papers)} papers from Scopus using refined query.")
    return papers

def get_all_values(d):
    if isinstance(d, dict):
        if "ORCID" in d:
            d.pop("ORCID")
        for v in d.values():
            yield from get_all_values(v)
    elif isinstance(d, Iterable) and not isinstance(d, str): # or list, set, ... only
        for v in d:
            yield from get_all_values(v)
    else:
        yield d 

def search_springer(query, max_results):
    base_url = "https://api.springernature.com/openaccess/json"
    params = {
        "api_key": SPRINGER_API_KEY,
        "q": query,
        "p": max_results
    }
    try:
        response = requests.get(base_url, params=params, timeout=30)
        print(f"Springer URL: {response.url}")
        # Check the response status code
        if response.status_code != 200:
            print(f"Springer API returned status code {response.status_code}: {response.text}")
            return []
        data = response.json()
    except Exception as e:
        print(f"Error retrieving or parsing Springer response: {e}")
        return []
    
    records = data.get("records", [])
    papers = []
    for record in records:
        title = record.get("title", "N/A")
        doi = record.get("doi", "N/A")
        year = record.get("publicationDate", "")
        if doi == "N/A" or not doi:
            continue  # Only process records with a DOI.
        doi_url = f"https://doi.org/{doi}"
        
        abstract = record.get("abstract", "")
        
        abstract_text = ""
        try:
            if isinstance(abstract, str):
                abstract = abstract.strip()
                if abstract.startswith("{") or abstract.startswith("["):
                    try:
                        abstract = json.loads(abstract)
                    except Exception as e:
                        print(f"Error processing Springer abstract for '{title}': {e}")
                        
            if isinstance(abstract, list) and len(abstract) > 0:
                inner = abstract[0]
                if isinstance(inner, str):
                    try:
                        inner_obj = ast.literal_eval(inner)
                    except Exception as e:
                        print(f"Error literal_eval for '{title}': {e}")
                        inner_obj = inner  # fallback to raw string
                else:
                    inner_obj = inner
                if isinstance(inner_obj, dict):
                    p_val = inner_obj.get("p", "")
                    if isinstance(p_val, list):
                        paragraphs = []
                        for item in p_val:
                            if isinstance(item, str):
                                text_item = item.strip()
                                if text_item.lower() in {"abstract", "abstract:"}:
                                    continue
                                paragraphs.append(text_item)
                        abstract_text = " ".join(paragraphs)
                    elif isinstance(p_val, str):
                        abstract_text = p_val.strip()
                    else:
                        abstract_text = str(p_val)
                else:
                    abstract_text = str(inner_obj)
            elif isinstance(abstract, dict):
                p_val = abstract.get("p", "")
                if isinstance(p_val, list):
                    paragraphs = []
                    for item in p_val:
                        if isinstance(item, str):
                            text_item = item.strip()
                            if text_item.lower() in {"abstract", "abstract:"}:
                                continue
                            paragraphs.append(text_item)
                    abstract_text = " ".join(paragraphs)
                elif isinstance(p_val, str):
                    abstract_text = p_val.strip()
                else:
                    abstract_text = str(p_val)
            else:
                abstract_text = str(abstract)
        except Exception as e:
            print(f"Error extracting Springer abstract text for '{title}': {e}")
            abstract_text = str(abstract)
        
        if not normalise_text(abstract_text):
            print(f"Skipping record with empty abstract: {title}")
            continue
        
        authors = "No Authors Available"
        creators = record.get("creators", [])
        if creators and isinstance(creators, list):
            author_names = [creator.get("creator", "").strip() for creator in creators if creator.get("creator", "").strip()]
            if author_names:
                authors = ", ".join(author_names)
        
        source = record.get("publicationName", record.get("publisher", "N/A")) + " (Springer)"
        
        paper = {
            "Title": normalise_text(title),
            "Source": normalise_text(source),
            "DOI": normalise_text(doi_url),
            "doi_suffix": doi,
            "Abstract": normalise_text(abstract_text),
            "Authors": normalise_text(authors),
            "Year": year
        }
        papers.append(paper)
    
    print(f"Found {len(papers)} papers from Springer.")
    return papers

def scrape_abstract_from_doi(doi_url, tr, title=None, retries=2, timeout=30):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AbstractScraper/1.0)"}
    for attempt in range(retries):
        try:
            response = tr.get(doi_url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                abstract_text = None  # initialise

                if response.url.startswith("https://dl.acm"):
                    acm_section = soup.find(id="abstract")
                    print(f"Scraping ACM abstract for DOI: {doi_url}")
                    if acm_section:
                        abstract_text = acm_section.get_text(strip=True).replace("Abstract", "", 1)

                if abstract_text is None and "sciencedirect.com" in response.url:
                    print("Scraping ScienceDirect abstract:", doi_url)
                    sd_section = soup.find("div", class_=re.compile("Abstracts", re.IGNORECASE))
                    if sd_section:
                        abstract_text = sd_section.get_text(strip=True).replace("Abstract", "", 1)
                    if abstract_text is None:
                        sd_section = soup.find(id="abstracts")
                        if sd_section:
                            abstract_text = sd_section.get_text(strip=True).replace("Abstract", "", 1)
                    if abstract_text is None:
                        sd_section = soup.find("div", class_=re.compile("abstract", re.IGNORECASE))
                        if sd_section:
                            abstract_text = sd_section.get_text(strip=True).replace("Abstract", "", 1)

                if abstract_text is None and "mdpi.com" in response.url:
                    print("Scraping MDPI abstract:", doi_url)
                    mdpi_section = soup.find("section", class_="html-abstract")
                    if mdpi_section:
                        abstract_text = mdpi_section.get_text(strip=True).replace("Abstract", "", 1)

                if abstract_text is None and "ieeexplore.ieee.org" in response.url:
                    print("Scraping IEEE abstract:", doi_url)
                    ieee_section = soup.find("div", class_="abstract-text")
                    if ieee_section:
                        abstract_text = ieee_section.get_text(strip=True).replace("Abstract", "", 1)

                if abstract_text is None:
                    print(f"Scraping generic abstract for DOI: {doi_url}")
                    meta_abstract = soup.find("meta", {"name": "citation_abstract"})
                    if meta_abstract and meta_abstract.get("content"):
                        abstract_text = meta_abstract["content"].strip()
                    if abstract_text is None:
                        abstract_elem = soup.find(class_=re.compile("abstract", re.IGNORECASE))
                        if abstract_elem:
                            abstract_text = abstract_elem.get_text(strip=True)
                    if abstract_text is None:
                        id_elem = soup.find(id="Abs1-content")
                        if id_elem:
                            abstract_text = id_elem.get_text(strip=True)

                if abstract_text:
                    return abstract_text
                else:
                    print(f"No abstract found in page for {doi_url}")
                    break 
            else:
                print(f"Unexpected status code {response.status_code} for {doi_url}")
                break
        except requests.exceptions.ReadTimeout:
            print(f"Read timeout on attempt {attempt+1} for {doi_url}. Skipping this paper.")
            return None  # Immediately return None so the caller can skip this paper.
        except Exception as e:
            print(f"Error scraping DOI page {doi_url}: {e}")
            break
    return None
