
from turtle import down
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
import os
from urllib.parse import urljoin
from urllib.request import urlretrieve
import itertools

ACL_URL = "https://aclanthology.org"
VENUE_SUFFIX = "/venues/"
MAX_PDF_COUNT = 200
DOWNLOADS_DIR = "data/"

def parse_acl_home_page():
    pdf_count = 0
    html_doc = requests.get(ACL_URL)
    soup = BeautifulSoup(html_doc.content, 'html.parser')
    
    tables = soup.findAll("tbody")
    # for table in tables:
    # Get only ACL events
    
    tags = tables[0].findAll("a")
    links = list(map(lambda tag: tag.get("href"), tags))
    os.makedirs(DOWNLOADS_DIR, exist_ok=False)
    
    venue_links = list(filter(lambda link: link.startswith(VENUE_SUFFIX), links))
    for link in venue_links:
        venue_link = urljoin(ACL_URL, link)
        venue_content = requests.get(venue_link).content
        venue_soup = BeautifulSoup(venue_content, 'html.parser', parse_only=SoupStrainer('li'))
        anchor_tags = list(itertools.chain.from_iterable([li.findAll('a') for li in venue_soup]))
        proceeding_links = [a.get("href") for a in anchor_tags if a.get("href").startswith("/volumes/")]
        # TODO: yield links here

        for proceeding_link in proceeding_links:
            proceeding_content = requests.get(urljoin(ACL_URL, proceeding_link)).content
            proceeding_soup = BeautifulSoup(proceeding_content, 'html.parser')
            button = proceeding_soup.select("button#toggle-all-abstracts")[0]
            paper_link_spans = button.find_next("div").find_all_next("strong")
            link_anchor_tags = [span.find_all("a") for span in paper_link_spans]
            paper_links = []
            
            import pdb;  pdb.set_trace()
            for anchor in link_anchor_tags:
                # paper_links.append(anchor[0].get("href"))
                if pdf_count > MAX_PDF_COUNT:
                    return 
                pdf_link = anchor[0].get("href").rstrip("/") + ".pdf"
                download_link = urljoin(ACL_URL, pdf_link)
                paper_links.append(download_link)
                urlretrieve(download_link, os.path.join(DOWNLOADS_DIR, pdf_link.strip("/")))
                pdf_count+=1
                # r = requests.get(download_link, stream=True)

                # with open(pdf_link, 'wb') as fd:
                #     for chunk in r.iter_content(chunk_size):
                #         fd.write(chunk)   
            # paper_links = [anchor[0].get("href", None) for anchor in link_anchor_tags if anchor.get("href").endswith(".pdf")]
            print(paper_links)
            
            
        
        
    print(event_links)
    # tags = tables[0].findAll(lambda tag: tag.name=='tr')
    # tds = list(map(lambda tag: tag.findAll("td"), tags))

parse_acl_home_page()