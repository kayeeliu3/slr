#!/usr/bin/env python3
import csv
import io
import shelve
import json, os, re, uuid, requests, time
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for, session
from flask_session import Session
from dotenv import load_dotenv
from google import genai
from unidecode import unidecode
from markdown import markdown

from app import search_literature, refine_db_queries
from db import index_papers, query_vector_db

app = Flask(__name__)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

load_dotenv()
app.secret_key = os.getenv("FLASK_KEY", "fallback-key")

SHELVE_FILENAME = 'cache.db'

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
SPRINGER_API_KEY = os.getenv('SPRINGER_API_KEY')
SCOPUS_API_KEY = os.getenv('SCOPUS_API_KEY')

MENDELEY_CLIENT_ID = os.getenv("MENDELEY_CLIENT_ID")
MENDELEY_CLIENT_SECRET = os.getenv("MENDELEY_CLIENT_SECRET")
MENDELEY_REDIRECT_URI = os.getenv("MENDELEY_REDIRECT_URI")
MENDELEY_AUTH_URL = "https://api.mendeley.com/oauth/authorize"
MENDELEY_TOKEN_URL = "https://api.mendeley.com/oauth/token"

# Helper functions for caching vector results
def save_to_shelve(data, shelf_filename='cache.db'):
    key = str(uuid.uuid4())
    with shelve.open(shelf_filename) as shelf:
        shelf[key] = data
    return key

def load_from_shelve(key, shelf_filename='cache.db'):
    with shelve.open(shelf_filename) as shelf:
        return shelf.get(key)

def delete_from_shelve(key, shelf_filename='cache.db'):
    with shelve.open(shelf_filename) as shelf:
        if key in shelf:
            del shelf[key]

@app.template_filter('md')
def markdown_filter(text):
    return markdown(text) # In the case of Gemini markdown outputs

@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        additional_keywords = request.form.get("additional_keywords", "")
        enabled_dbs = []
        if request.form.get("pubmed"):
            enabled_dbs.append("pubmed")
        if request.form.get("europe"):
            enabled_dbs.append("europe")
        if request.form.get("scopus"):
            enabled_dbs.append("scopus")
        if request.form.get("springer"):
            enabled_dbs.append("springer")
        search_method = request.form.get("search_method")
        
        # Store pending values in session, including additional keywords
        session["pending_query"] = query
        session["pending_additional_keywords"] = additional_keywords
        session["pending_enabled_dbs"] = enabled_dbs
        session["pending_search_method"] = search_method
        
        # Redirect to the confirmation page
        return redirect(url_for("confirm"))
            
    return render_template("index.html")

@app.route("/confirm", methods=["GET", "POST"])
def confirm():
    if request.method == "POST":
        query = session.get("pending_query")
        enabled_dbs = session.get("pending_enabled_dbs")
        search_method = session.get("pending_search_method")
        
        session.pop("refined_queries", None)
        session.pop("pending_query", None)
        session.pop("pending_additional_keywords", None)
        session.pop("pending_enabled_dbs", None)
        session.pop("pending_search_method", None)
        
        return redirect(url_for(
            f"{search_method}_results",
            query=query,
            pubmed="on" if "pubmed" in enabled_dbs else None,
            europe="on" if "europe" in enabled_dbs else None,
            scopus="on" if "scopus" in enabled_dbs else None,
            springer="on" if "springer" in enabled_dbs else None
        ))
    else:
        query = session.get("pending_query")
        enabled_dbs = session.get("pending_enabled_dbs")
        additional_keywords = session.get("pending_additional_keywords", "")
        
        if not query or not enabled_dbs:
            return redirect(url_for("index"))
        
        session.pop("refined_queries", None) # Clear
        refined_queries = refine_db_queries(query, additional_keywords)
        session["refined_queries"] = refined_queries
        
        filtered_refined = {db: refined_queries[db] for db in enabled_dbs if db in refined_queries}
        return render_template("confirm.html", refined_queries=filtered_refined, query=query, additional_keywords=additional_keywords)

@app.route("/vector-results")
def vector_results():
    query = request.args.get("query")
    enabled_dbs = []
    if request.args.get("pubmed") is not None:
        enabled_dbs.append("pubmed")
    if request.args.get("europe") is not None:
        enabled_dbs.append("europe")
    if request.args.get("scopus") is not None:
        enabled_dbs.append("scopus")
    if request.args.get("springer") is not None:
        enabled_dbs.append("springer")
        
    query_results = run_vector_query(query, query, enabled_dbs)
    refined_queries = session.get("refined_queries", {})
    
    # Store vector results into session for CSV export
    cache_key = save_to_shelve(query_results)
    session["vector_results_key"] = cache_key

    return render_template("vector_results.html", results=query_results, query=query, total=len(query_results), refined_queries=refined_queries)

@app.route("/llm-results")
def llm_results():
    query = request.args.get("query")
    enabled_dbs = []
    if request.args.get("pubmed") is not None:
        enabled_dbs.append("pubmed")
    if request.args.get("europe") is not None:
        enabled_dbs.append("europe")
    if request.args.get("scopus") is not None:
        enabled_dbs.append("scopus")
    if request.args.get("springer") is not None:
        enabled_dbs.append("springer")
        
    query_results = run_gemini_query(query, enabled_dbs)
    refined_queries = session.get("refined_queries", {})
    
    cache_key = save_to_shelve(query_results)
    session["llm_results_key"] = cache_key

    return render_template("llm_results.html", results=query_results, query=query, total=len(query_results), refined_queries=refined_queries)

@app.route("/export_csv")
def export_csv():
    # Look for the cached key in session; try LLM results first, then vector
    cache_key = session.get("llm_results_key") or session.get("vector_results_key")
    if not cache_key:
         return "No search results available to export.", 400

    search_results = load_from_shelve(cache_key)
    if not search_results:
         return "No cached search results found.", 400

    output = io.StringIO()
    fieldnames = ["source", "Year", "title", "doi", "authors"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for result in search_results:
        writer.writerow({
           "source": result.get("source", ""),
           "Year": result.get("Year", ""),
           "title": result.get("title", ""),
           "doi": result.get("doi", ""),
           "authors": result.get("authors", "")
        })
    csv_data = output.getvalue()
    output.close()
    
    delete_from_shelve(cache_key)

    return Response(csv_data, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=search_results.csv"})

@app.route("/extraction")
def extraction_results():
    db = request.args.get("db")
    extraction_data = {}
    
    if db == "pubmed":
        pmcid = request.args.get("pmcid")
        doi = request.args.get("doi")
        title = request.args.get("title")
        if not pmcid:
            return "PMCID not provided.", 400
        extraction_data = extract_data_from_full_text("pubmed", pmcid=pmcid, doi=doi, title=title)
    elif db == "springer":
        doi = request.args.get("doi")
        title = request.args.get("title")
        if not doi:
            return "DOI not provided.", 400
        extraction_data = extract_data_from_full_text("springer", doi=doi, title=title)
    elif db == "scopus":
        doi = request.args.get("doi")
        title = request.args.get("title")
        if not doi:
            return "DOI not provided.", 400
        extraction_data = extract_data_from_full_text("scopus", doi=doi, title=title)
    elif db == "europe":
        europe_pmc = request.args.get("pmcid")
        doi = request.args.get("doi")
        title = request.args.get("title")
        if not europe_pmc:
            return "Europe PMC ID not provided.", 400
        extraction_data = extract_data_from_full_text("europe", pmcid=europe_pmc, doi=doi, title=title)
    else:
        return "Unsupported database.", 400
    
    # If extraction_data is not a dict, assume it is a redirect (manual extraction fallback)
    if not isinstance(extraction_data, dict):
        return extraction_data

    return render_template("extraction.html", extraction=extraction_data)

@app.route("/extraction-text", methods=["GET", "POST"])
def extraction_text():
    if request.method == "POST":
        full_text = request.form.get("full_text")
        title = request.form.get("title", "No Title Provided")
        doi = request.form.get("doi", "No DOI Provided")
        
        if not full_text or not full_text.strip():
            return "No text provided", 400
        
        extraction = gemini_extract_data(full_text)
        if extraction.get("summary", "").strip() == "N/A":
            extraction_result = {
                "statistics": extraction.get("statistics", []),
                "key_points": extraction.get("key_points", []),
                "summary": "Unable to extract data from the provided text."
            }
        else:
            extraction_result = {
                "statistics": extraction.get("statistics", []),
                "key_points": extraction.get("key_points", []),
                "summary": extraction.get("summary")
            }
        return render_template("extraction.html", extraction=extraction_result)
    
    # GET Request
    extraction_data = {
        "title": request.args.get("title", "No Title Provided"),
        "doi": request.args.get("doi", "No DOI Provided")
    }
    return render_template("extraction_text.html", extraction=extraction_data)

@app.route("/compare_extractions", methods=["POST"])
def compare_extractions():
    # Retrieve the JSON-encoded results from the form.
    results_json = request.form.get("results")
    if not results_json:
        return "No results provided", 400
    try:
        results_list = json.loads(results_json)
        print(results_list)
    except Exception as e:
        return f"Error parsing results: {str(e)}", 400

    extraction_results = []
    for paper in results_list:
        # Determine which database and identifiers to use based on the source.
        source = paper.get("source", "").lower()
        try:
            if "pubmed" in source:
                db_type = "pubmed"
                pmcid = paper.get("PMCID")
                print(f"Attempting PUBMED fetch of PMCID: {pmcid}")
                if not pmcid or pmcid == "N/A":
                    print("Skipping PubMed...")
                    continue
                full_text = fetch_full_text("pubmed", pmcid=pmcid)
            elif "springer" in source:
                db_type = "springer"
                doi = paper.get("doi_suffix") or paper.get("DOI")
                print(f"Attempting Springer fetch of DOI: {doi}")
                if not doi or doi == "N/A":
                    print("Skipping Springer...")
                    continue
                full_text = fetch_full_text("springer", doi=doi)
            elif "scopus" in source:
                db_type = "scopus"
                doi = paper.get("doi_suffix") or paper.get("DOI")
                print(f"Attempting Scopus fetch of DOI: {doi}")
                if not doi or doi == "N/A":
                    print("Skipping Scopus...")
                    continue
                full_text = fetch_full_text("scopus", doi=doi)
            elif "europe" in source:
                db_type = "europe"
                pmcid = paper.get("PMCID")
                print(f"Attempting Europe PMC fetch of DOI: {pmcid}")
                if not pmcid or pmcid == "N/A":
                    print("Skipping Europe PMC...")
                    continue
                full_text = fetch_full_text("europe", pmcid=pmcid)
            else:
                continue  # Unsupported source
        except Exception as e:
            # Skip this paper if full text cannot be fetched.
            print("Cannot fetch paper.")
            continue

        if not full_text.strip() or "N/A" in full_text:
            print("Full text is not available.")
            continue  # Skip if the full text is empty or marked unavailable.

        # Use your extraction function to process the full text.
        time.sleep(3)
        extraction = gemini_extract_data(full_text)
        # Include the paper title (for later reference in comparative insights)
        extraction["title"] = paper.get("title") or paper.get("Title", "Unknown Title")
        extraction_results.append(extraction)
    
    if not extraction_results:
        return "No papers had available full text for extraction.", 400

    # Aggregate the extraction outputs into one prompt.
    aggregated_text = "Comparative Extraction Data:\n\n"
    for ext in extraction_results:
        aggregated_text += f"Paper: {ext.get('title')}\n"
        aggregated_text += "Statistics:\n"
        for stat in ext.get("statistics", []):
            aggregated_text += f"- {stat.get('statistic', '')}: {stat.get('context', '')}\n"
        aggregated_text += "Key Points:\n"
        for kp in ext.get("key_points", []):
            aggregated_text += f"- {kp}\n"
        aggregated_text += f"Summary: {ext.get('summary', '')}\n\n"
    
    prompt = (
        "You are given aggregated extraction data from multiple research papers. Each extraction includes the paper's title, statistics, key points, and summary. "
        "Assign a unique reference number to each paper in the aggregated data (for example, [1], [2], [3], etc.) and include a 'references' field in your output that maps each reference number to the corresponding paper title. "
        "Then, based on the aggregated data below, generate comparative insights in JSON format with the following keys:\n"
        "  - 'references': a mapping of reference numbers to paper titles.\n"
        "  - 'similarities': a very detailed description of the similarities observed across the papers, referring to them by their reference numbers (e.g., [1], [2]).\n"
        "  - 'differences': a very detailed description of the differences observed across the papers, referring to them by their reference numbers.\n"
        "  - 'correlations': a very description of any correlations or relationships between the statistics and key points, with references to the papers by their numbers.\n"
        "  - 'summary': an overall brief summary of the comparative insights.\n\n"
        "The output must be valid JSON with the keys exactly as specified. For example:\n\n"
        "```\n"
        "{\n"
        "  \"references\": {\"[1]\": \"Paper Title 1\", \"[2]\": \"Paper Title 2\", \"[3]\": \"Paper Title 3\"},\n"
        "  \"similarities\": \"[1] and [2] both indicate ...\",\n"
        "  \"differences\": \"[1] emphasises ... while [3] shows ...\",\n"
        "  \"correlations\": \"Key points in [2] correlate with the statistics in [3] ...\",\n"
        "  \"summary\": \"Overall, the research papers show ...\"\n"
        "}\n"
        "```\n\n"
        "Now, based on the following aggregated extraction data, generate the comparative insights (and only the insights itself):\n\n"
        + aggregated_text
    )

    
    print(prompt)

    print("Final Gemini query with aggregrated text...")
    # Call Gemini with the aggregated prompt.
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=prompt
    )
    
    try:
        insights = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        print(insights)
    except Exception as e:
        # In case parsing fails, fallback to a simple error object.
        insights = {
            "error": f"Error parsing JSON: {str(e)}",
            "raw": response.text
        }

    return render_template("compare_extractions.html", insights=insights)

@app.route("/manual_comparison", methods=["GET", "POST"])
def manual_comparison():
    if request.method == "POST":
        paper_titles = request.form.getlist("paper_title")
        full_texts = request.form.getlist("full_text")
        
        if not full_texts or all(not txt.strip() for txt in full_texts):
            return "No text provided", 400

        extraction_results = []
        for title, text in zip(paper_titles, full_texts):
            if text and text.strip():
                time.sleep(3)
                extraction = gemini_extract_data(text)
                extraction["title"] = title.strip() if title.strip() else "Manual Entry"
                extraction_results.append(extraction)
        
        if not extraction_results:
            return "No valid texts provided for extraction.", 400

        aggregated_text = "Comparative Extraction Data:\n\n"
        for ext in extraction_results:
            aggregated_text += f"Paper: {ext.get('title')}\n"
            aggregated_text += "Statistics:\n"
            for stat in ext.get("statistics", []):
                aggregated_text += f"- {stat.get('statistic', '')}: {stat.get('context', '')}\n"
            aggregated_text += "Key Points:\n"
            for kp in ext.get("key_points", []):
                aggregated_text += f"- {kp}\n"
            aggregated_text += f"Summary: {ext.get('summary', '')}\n\n"
        
        # Updated prompt: force the output to include a "references" field
        prompt = (
            "You are given aggregated extraction data from multiple research papers. Each extraction includes the paper's title, statistics, key points, and summary. "
            "Assign a unique reference number to each paper in the aggregated data (for example, [1], [2], [3], etc.) and include a 'references' field in your output that maps each reference number to the corresponding paper title. "
            "Then, based on the aggregated data below, generate comparative insights in JSON format with the following keys:\n"
            "  - 'references': a mapping of reference numbers to paper titles.\n"
            "  - 'similarities': a very detailed description of the similarities observed across the papers, referring to them by their reference numbers (e.g., [1], [2]).\n"
            "  - 'differences': a very detailed description of the differences observed across the papers, referring to them by their reference numbers.\n"
            "  - 'correlations': a description of any correlations or relationships between the statistics and key points, with references to the papers by their numbers.\n"
            "  - 'summary': an overall brief summary of the comparative insights.\n\n"
            "The output must be valid JSON with the keys exactly as specified. For example:\n\n"
            "```\n"
            "{\n"
            "  \"references\": {\"[1]\": \"Paper Title 1\", \"[2]\": \"Paper Title 2\", \"[3]\": \"Paper Title 3\"},\n"
            "  \"similarities\": \"[1] and [2] both indicate ...\",\n"
            "  \"differences\": \"[1] emphasises ... while [3] shows ...\",\n"
            "  \"correlations\": \"Key points in [2] correlate with the statistics in [3] ...\",\n"
            "  \"summary\": \"Overall, the research papers show ...\"\n"
            "}\n"
            "```\n\n"
            "Now, based on the following aggregated extraction data, generate the comparative insights (and only the insights itself):\n\n"
            + aggregated_text
        )
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=prompt
        )
        
        try:
            insights = json.loads(response.text.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            insights = {
                "error": f"Error parsing JSON: {str(e)}",
                "raw": response.text
            }
        
        return render_template("compare_extractions.html", insights=insights)
    else:
        return render_template("manual_comparison.html")


# Helper function to parse names into "first_name" "last_name" for Mendeley export
def parse_authors(authors_str):
    # Use semicolon delimiter if present
    if ";" in authors_str:
        author_entries = [entry.strip() for entry in authors_str.split(";") if entry.strip()]
    else:
        # Otherwise, check if the string is in "Lastname, Firstname" format.
        # If there is at least one comma inside each author segment, we assume that.
        # For example, if the string looks like "Smith, John; Doe, Jane", we expect a comma.
        # But if there are no semicolons, sometimes authors are simply separated by commas.
        # In that case, we try to pair adjacent items if the number of splits is even.
        parts = [part.strip() for part in authors_str.split(",") if part.strip()]
        if len(parts) % 2 == 0:
            # Assume that each pair is LastName followed by FirstName.
            author_entries = []
            for i in range(0, len(parts), 2):
                # Combine as "FirstName LastName"
                author_entries.append(f"{parts[i+1]} {parts[i]}")
        else:
            # Otherwise, assume names are separated by commas and split on whitespace.
            author_entries = [entry.strip() for entry in authors_str.split(",") if entry.strip()]
    
    authors_list = []
    for entry in author_entries:
        if not entry:
            continue
        # Try to detect if entry is in "Lastname, Firstname" format:
        if "," in entry:
            subparts = [sub.strip() for sub in entry.split(",")]
            last_name = subparts[0] if subparts[0] else "Unknown"
            first_name = subparts[1] if len(subparts) > 1 and subparts[1] else "Unknown"
        else:
            parts = entry.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = " ".join(parts[1:]).strip()
            else:
                first_name = parts[0]
                last_name = "Unknown"
        authors_list.append({
            "first_name": first_name if first_name else "Unknown",
            "last_name": last_name if last_name else "Unknown"
        })
    return authors_list



@app.route("/mendeley/export")
def mendeley_export():
    access_token = session.get("mendeley_token")
    
    doi = request.args.get("doi")
    title = request.args.get("title")
    authors_str = request.args.get("authors")
    
    if not doi or not title or not authors_str:
        export_data = session.get("mendeley_export_data", {})
        doi = doi or export_data.get("doi")
        title = title or export_data.get("title")
        authors_str = authors_str or export_data.get("authors")
    
    authors_data = parse_authors(authors_str) if authors_str else []
    
    if not access_token:
        session["mendeley_export_data"] = {
            "doi": doi,
            "title": title,
            "authors": authors_str
        }
        return redirect(url_for("mendeley_link"))
    
    citation_payload = {
        "title": title,
        "authors": authors_data,
        "identifiers": {
            "doi": doi
        },
        "type": "generic"
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/vnd.mendeley-document.1+json",
        "Content-Disposition": 'attachment; filename="citation.json"'
    }
    
    response = requests.post("https://api.mendeley.com/documents", headers=headers, json=citation_payload)
    
    if response.status_code == 401:  # Expired token/Unauthorised.
        session.pop("mendeley_token", None)
        session["mendeley_export_data"] = {
            "doi": doi,
            "title": title,
            "authors": authors_str
        }
        return redirect(url_for("mendeley_link"))
    
    if response.status_code == 201:
        status = "success"
        message = "Export successful."
    else:
        status = "error"
        message = "Failed to export citation: " + response.text.replace("\n", " ") + ". Please try again."
    
    html = f"""
    <html>
      <body>
        <script>
          if (window.opener && !window.opener.closed) {{
            window.opener.postMessage({{'status': '{status}', 'message': '{message}'}}, "*");
          }}
          window.close();
        </script>
        <p>{message} You can close this window.</p>
      </body>
    </html>
    """
    return html

@app.route("/mendeley/export_all", methods=["GET", "POST"])
def mendeley_export_all():
    access_token = session.get("mendeley_token")
    # If not authenticated, store citations in session and redirect for auth.
    if not access_token:
        # For GET requests, citations should be passed in the URL.
        # If not found, also check session.
        citations_param = request.args.get("citations")
        if citations_param:
            try:
                citations = json.loads(citations_param)
            except Exception as e:
                return jsonify({"error": "Failed to decode citations: " + str(e)}), 400
        else:
            citations = None
        
        if not citations:
            # If no citations in URL, assume they are being exported now and store an empty list (or use your existing export all data)
            # You need to decide how to pass the citations from your client.
            return jsonify({"error": "No citations provided"}), 400
        
        # Store the citations in session for later export
        session["mendeley_export_all_data"] = citations
        # Redirect to authentication and pass next=export_all
        return redirect(url_for("mendeley_link", next="export_all"))
    
    # For authenticated users, try to get citations either from URL (for GET) or from session.
    if request.method == "GET":
        citations_param = request.args.get("citations")
        if citations_param:
            try:
                citations = json.loads(citations_param)
            except Exception as e:
                return jsonify({"error": "Failed to decode citations: " + str(e)}), 400
        else:
            citations = session.get("mendeley_export_all_data")
            if not citations:
                return jsonify({"error": "No citations provided"}), 400
    else:  # POST
        citations = request.get_json()
        if not citations:
            return jsonify({"error": "No citations provided"}), 400

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/vnd.mendeley-document.1+json"
    }
    
    # Process each citation. (Note: the loop below should really accumulate results.
    # Here we simply send back the response for the last citation.)
    for citation in citations:
        doi = citation.get("doi")
        title = citation.get("title")
        authors = parse_authors(citation.get("authors", ""))
        payload = {
            "title": title,
            "authors": authors,
            "identifiers": {"doi": doi},
            "type": "generic"
        }
        
        response = requests.post("https://api.mendeley.com/documents", headers=headers, json=payload)
        
        if response.status_code == 401:  # Expired token/Unauthorized.
            session.pop("mendeley_token", None)
            session["mendeley_export_all_data"] = citations
            return redirect(url_for("mendeley_link", next="export_all"))
        
        # Optionally, collect successes and failures for each citation.
        # For simplicity, here we only check the last response.
        if response.status_code == 201:
            status = "success"
            message = "Export successful."
        else:
            status = "error"
            message = "Failed to export citation: " + response.text.replace("\n", " ") + ". Please try again."
    
    # Clear the export-all data from session after successful export (optional).
    session.pop("mendeley_export_all_data", None)
    
    html = f"""
    <html>
    <body>
        <script>
        if (window.opener && !window.opener.closed) {{
            window.opener.postMessage({{'status': '{status}', 'message': '{message}'}}, "*");
        }}
        window.close();
        </script>
        <p>{message} You can close this window.</p>
    </body>
    </html>
    """
    return html


@app.route("/mendeley/link")
def mendeley_link():
    state = str(uuid.uuid4())
    session["mendeley_state"] = state
    # If a next parameter is provided, store it in session
    next_endpoint = request.args.get("next")
    if next_endpoint:
        session["next"] = next_endpoint
    params = {
        "client_id": MENDELEY_CLIENT_ID,
        "redirect_uri": MENDELEY_REDIRECT_URI,
        "response_type": "code",
        "scope": "all",
        "state": state
    }
    auth_url = requests.Request('GET', MENDELEY_AUTH_URL, params=params).prepare().url
    print(f"Redirecting to auth: {auth_url}")
    return redirect(auth_url)

@app.route("/mendeley/callback")
def mendeley_callback():
    """
    Handles the OAuth callback.
    After obtaining the access token, if export data exists, automatically retries export.
    Otherwise, it sends a success message and closes the window.
    """
    code = request.args.get("code")
    state = request.args.get("state")
    if state != session.get("mendeley_state"):
        return "State mismatch. Please try again.", 400

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": MENDELEY_REDIRECT_URI,
        "client_id": MENDELEY_CLIENT_ID,
        "client_secret": MENDELEY_CLIENT_SECRET
    }
    token_response = requests.post(MENDELEY_TOKEN_URL, data=data)
    if token_response.status_code != 200:
        return "Error obtaining access token.", token_response.status_code

    token_data = token_response.json()
    session["mendeley_token"] = token_data.get("access_token")
    
    if "mendeley_export_data" in session:
        # Retry export by redirecting to /mendeley/export in this same tab.
        return redirect(url_for("mendeley_export"))
    else:
        # If no export data, simply notify the parent window.
        html = """
        <html>
          <body>
            <script>
              if (window.opener && !window.opener.closed) {
                window.opener.postMessage({status: 'success', message: 'Mendeley authentication successful.'}, "*");
              }
              window.close();
            </script>
            <p>Authentication successful! You can close this window.</p>
          </body>
        </html>
        """
        return html@app.route("/mendeley/callback")

def mendeley_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    if state != session.get("mendeley_state"):
        return "State mismatch. Please try again.", 400

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": MENDELEY_REDIRECT_URI,
        "client_id": MENDELEY_CLIENT_ID,
        "client_secret": MENDELEY_CLIENT_SECRET
    }
    token_response = requests.post(MENDELEY_TOKEN_URL, data=data)
    if token_response.status_code != 200:
        return "Error obtaining access token.", token_response.status_code

    token_data = token_response.json()
    session["mendeley_token"] = token_data.get("access_token")
    
    # Check for the "next" endpoint stored in session:
    next_endpoint = session.pop("next", None)
    if next_endpoint == "export_all":
        # Redirect to the export all endpoint.
        return redirect(url_for("mendeley_export_all"))
    elif "mendeley_export_data" in session:
        # For single citation export.
        return redirect(url_for("mendeley_export"))
    else:
        # Otherwise, just notify success.
        html = """
        <html>
          <body>
            <script>
              if (window.opener && !window.opener.closed) {
                window.opener.postMessage({status: 'success', message: 'Mendeley authentication successful.'}, "*");
              }
              window.close();
            </script>
            <p>Authentication successful! You can close this window.</p>
          </body>
        </html>
        """
        return html

@app.route("/mendeley/export_retry")
def mendeley_export_retry():
    export_data = session.pop("mendeley_export_data", None)
    if not export_data:
        return redirect(url_for("index"))
    
    access_token = session.get("mendeley_token")
    if not access_token:
        return redirect(url_for("mendeley_link", next=export_data.get("next")))
    
    doi = export_data.get("doi")
    title = export_data.get("title")
    authors = export_data.get("authors")
    next_url = export_data.get("next") or url_for("index")
    
    # Check if authors is a list; if not, split the string.
    if isinstance(authors, list):
        authors_list = authors
    else:
        authors_list = authors.split(",") if authors else []
    
    citation_payload = {
        "title": title,
        "authors": authors_list,
        "identifiers": {
            "doi": doi
        },
        "type": "generic"
    }
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/vnd.mendeley-document.1+json",
        "Content-Disposition": 'attachment; filename="citation.json"'
    }
    
    response = requests.post("https://api.mendeley.com/documents", headers=headers, json=citation_payload)
    if response.status_code == 201:
        print("Export successful (retry).")
        return redirect(next_url)
    else:
        print("Export retry failed.")
        return jsonify({"status": "error", "message": "Failed to export citation", "details": response.text}), response.status_code

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
            "PMCID": metadatas[i].get("PMCID"),
            "Year": metadatas[i].get("Year")
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
        paper["Year"] = paper.get("Year")
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
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")
            
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
        f"The 'summary' key should be a text summary of the whole paper, including what its concluding remarks are.\n"
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


def extract_data_from_full_text(type, doi=None, pmcid=None, title=None):
    try:
        if type == "pubmed":
            full_text = fetch_full_text("pubmed", pmcid=pmcid)
            print(f"Full text acquired for PMC ID: {pmcid}.")
        elif type == "springer":
            full_text = fetch_full_text("springer", doi=doi)
            print(f"Full text acquired for Springer with DOI: {doi}.")
        elif type == "scopus":
            full_text = fetch_full_text("scopus", doi=doi)
            print(f"Full text acquired for Scopus with DOI: {doi}.")
        elif type == "europe":
            full_text = fetch_full_text("europe", pmcid=pmcid)
            print(f"Full text acquired for Europe PMC ID: {pmcid}.")
        else:
            return "Unsupported database.", 400
    except Exception as e:
        print("Error fetching full text:", e)
        # Pass title and doi if available to the manual extraction page.
        return redirect(url_for("extraction_text", title=title, doi=doi))
    
    if not full_text.strip() or "N/A" in full_text:
        print("Full text is not available or indicates unavailability.")
        return redirect(url_for("extraction_text", title=title, doi=doi))
    
    extraction = gemini_extract_data(full_text)
    if extraction.get("summary", "").strip() == "N/A":
        print("Extraction summary is N/A. Redirecting to manual extraction.")
        return redirect(url_for("extraction_text", title=title, doi=doi))
    
    # Process and aggregate extraction data as before...
    aggregated_statistics = []
    aggregated_key_points = []
    for res in [extraction]:
        stat = res.get("statistics", {})
        if isinstance(stat, list):
            aggregated_statistics.extend(stat)
        else:
            aggregated_statistics.append(stat)
        kp = res.get("key_points", [])
        if isinstance(kp, list):
            aggregated_key_points.extend(kp)
        else:
            aggregated_key_points.append(kp)
    
    return {"statistics": aggregated_statistics, "key_points": aggregated_key_points, "summary": extraction.get("summary")}

if __name__ == '__main__':
    app.run(debug=True)
