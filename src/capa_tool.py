import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID", "")


def _get_anthropic_key() -> str:
    """Resolve Anthropic API key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            for key_name in ["ANTHROPIC_API_KEY", "anthropic_api_key", "claude_api_key", "claude"]:
                if key_name in st.secrets:
                    val = str(st.secrets[key_name]).strip()
                    if val:
                        return val
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or ""

# ==========================================
# TOOL 1: US FDA DATABASE (Structured Data)
# ==========================================
def search_openfda(device_name, manufacturer=None):
    """
    Queries the official US FDA database for enforcement reports and recalls.
    """
    base_url = "https://api.fda.gov/device/enforcement.json"
    
    # Construct query
    query_parts = [f'product_description:"{device_name}"']
    if manufacturer:
        query_parts.append(f'recalling_firm:"{manufacturer}"')
    
    search_query = " AND ".join(query_parts)
    
    params = {
        'search': search_query,
        'limit': 5
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "error" in data:
            return "No specific FDA enforcement reports found in structured database."
            
        results = []
        for item in data.get('results', []):
            results.append({
                "source": "US FDA (Structured)",
                "recall_number": item.get('recall_number'),
                "reason": item.get('reason_for_recall'),
                "status": item.get('status'),
                "date": item.get('report_date')
            })
        return results
    except Exception as e:
        return f"Error querying OpenFDA: {str(e)}"

# ==========================================
# TOOL 2: GLOBAL SEARCH & MEDIA (Unstructured)
# ==========================================
def search_global_media_and_reg(query, category="general"):
    """
    Uses Google Search API to find global regulatory info (EU, UK, LATAM) and media.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Define site filters for regulatory bodies to ensure high-quality results
    regulatory_sites = [
        "site:gov.uk",           # UK MHRA
        "site:europa.eu",        # EU EMA/EUDAMED
        "site:anvisa.gov.br",    # Brazil ANVISA
        "site:cofepris.gob.mx",  # Mexico
        "site:fda.gov"           # US FDA (Web content)
    ]
    
    final_query = query
    
    # If looking for regulatory info specifically, boost official domains
    if category == "regulatory":
        site_string = " OR ".join(regulatory_sites)
        final_query = f"({site_string}) {query} recall OR alert OR warning OR safety"
    elif category == "media":
        final_query = f"{query} medical device recall news"

    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CX_ID,
        'q': final_query,
        'num': 5
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        results = []
        if 'items' in data:
            for item in data['items']:
                results.append({
                    "source": "Web/Media",
                    "title": item.get('title'),
                    "snippet": item.get('snippet'),
                    "link": item.get('link')
                })
        return results
    except Exception as e:
        return f"Error querying Google Search: {str(e)}"

# ==========================================
# MAIN APP CONTROLLER
# ==========================================
def analyze_device_safety(device_name, manufacturer=None, model=None):
    print(f"🔍 Starting CAPA investigation for: {device_name}...")
    
    # 1. Fetch Structured US Data
    print("   ...Querying US FDA Database")
    fda_data = search_openfda(device_name, manufacturer)
    
    # 2. Fetch Global Regulatory Data
    print("   ...Querying Global Regulatory Agencies (EU, UK, LATAM)")
    reg_query_str = f"{device_name} {model if model else ''} {manufacturer if manufacturer else ''}"
    global_reg_data = search_global_media_and_reg(reg_query_str, category="regulatory")
    
    # 3. Fetch Media/News Data
    print("   ...Querying News Media")
    media_data = search_global_media_and_reg(reg_query_str, category="media")
    
    # 4. Synthesize with AI
    print("   ...Synthesizing data with AI")
    
    prompt = f"""
    You are a Quality Assurance Regulatory Expert. Review the following data for the medical device:
    Device: {device_name}
    Manufacturer: {manufacturer}
    
    DATA SOURCES:
    1. US FDA Database: {json.dumps(fda_data)}
    2. Global Regulatory Web Hits: {json.dumps(global_reg_data)}
    3. Media News Hits: {json.dumps(media_data)}
    
    Please provide a CAPA (Corrective and Preventive Action) Risk Assessment Summary including:
    1. A summary of any active recalls (US or Global).
    2. Key safety warnings found in the media or regulatory bulletins.
    3. A categorization of the risk (Low/Medium/High) based on this data.
    4. Recommended next steps for the Quality team.
    """

    api_key = _get_anthropic_key()
    if not api_key:
        return "AI unavailable — add ANTHROPIC_API_KEY to Streamlit secrets."

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1500,
                "system": "You are a helpful medical device regulatory assistant.",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=45,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception as e:
        return f"AI Connection Failed: {e}"

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Example usage
    device = input("Enter Device Name (e.g., Pacemaker): ")
    mfg = input("Enter Manufacturer (optional): ")
    
    report = analyze_device_safety(device, mfg)
    
    print("\n" + "="*40)
    print("CAPA SAFETY REPORT")
    print("="*40)
    print(report)
