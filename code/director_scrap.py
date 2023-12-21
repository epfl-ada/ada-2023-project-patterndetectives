import requests
import re

def director_scrap(pageid):
    # Make the request
    r = requests.get(f'https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&pageids={pageid}&format=json') 

    # Convert the response to JSON
    data = r.json()

    # Check if page is missing
    if 'missing' in data['query']['pages'][str(pageid)]:
        return None

    # Check if revisions key exists
    if 'revisions' not in data['query']['pages'][str(pageid)]:
        return None

    # Extract the wikitext content of the page
    wikitext = data['query']['pages'][str(pageid)]['revisions'][0]['*']

    # Use a regular expression to extract the director's name
    pattern = r'\|\s*director\s*=\s*(?:(?:{{(?:ubl|Plainlist)\|)?\[\[)?([^\]|<\n\[{]+)'
    match = re.search(pattern, wikitext)

    if match:
        director = match.group(1).split('|')[-1].strip()

        # Clean up the name to remove any content within parentheses
        director = re.sub(r'\s*\([^)]+\)', '', director)

        if not director:  # Check if the extracted director name is empty
            return None

        return director
    else:
        return None
