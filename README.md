# slr
Automation aid for SLR implementations. To run, cd into src/app, and do `'python3 app.py'`.

Ensure for the tool the work, generate your own keys for the following APIs and plug into the .env file:

SCOPUS_API_KEY = "insert_key" (get from https://dev.elsevier.com)
SPRINGER_API_KEY = "insert_key" (get from https://dev.springernature.com)
GEMINI_API_KEY = "insert_key" (get from https://ai.google.dev/gemini-api/docs/api-key)

MENDELEY_CLIENT_ID = "insert_key" (get from https://dev.mendeley.com)
MENDELEY_CLIENT_SECRET = "insert_key"
MENDELEY_REDIRECT_URI = "http://127.0.0.1:5000/mendeley/callback"

FLASK_KEY = "insert_key" (can be any series of characters)

