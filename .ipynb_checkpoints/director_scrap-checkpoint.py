import requests
from bs4 import BeautifulSoup
import re
import numpy as np

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
    match = re.search(r'\| director\s*=\s*\[\[([^\]]+)\]\]', wikitext)
    if match:
        director = match.group(1)
        # Split the director string on '|' and take the latter part if '|' exists.
        director = director.split('|')[-1]
        print(f"Director of the movie is: {director}")
        return director
    else:
        print(f"Director not found")
        return None