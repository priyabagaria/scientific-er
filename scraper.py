
from turtle import down
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
import os
from urllib.parse import MAX_CACHE_SIZE, urljoin
from urllib.request import urlretrieve
import itertools

ACL_URL = "https://aclanthology.org"
VENUE_SUFFIX = "/venues/"
MAX_PDF_COUNT = 100
DOWNLOADS_DIR = "data/"
START_YEAR = 22
END_YEAR = 0
PAPERS_PER_PROCEEDING = 20

def retrieve_url(url):
    html_doc = requests.get(url)
    return BeautifulSoup(html_doc.content, 'html.parser')


def download_pdf(url):
    if url.startswith(ACL_URL):
        urlretrieve(url, os.path.join(DOWNLOADS_DIR, url.rsplit('/', 1)[-1]))
        return True
    return False

def parse_acl():
    """
        Parse home page by year
    """
    pdf_count = 0

    soup = retrieve_url(ACL_URL)

    acl_table = soup.find("table")
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    for year in range(START_YEAR, END_YEAR-1, -1):
        # acl_table.findAll("a", {"text": str(year).zfill(2)})
        venues = acl_table.findAll('a', href=True, text=str(year).zfill(2))
        venue_hrefs = [venue['href'] for venue in venues]
        for venue_href in venue_hrefs:
            venue_soup = retrieve_url(urljoin(ACL_URL, venue_href))
            print(venue_href)
            buttons = venue_soup.select("button#toggle-all-abstracts")
            if buttons == []:
                proceedings = venue_soup.find_all_next("div", attrs={'class' : None}, recursive=False)
            else:
                proceedings = buttons[0].find_all_next("div", attrs={'class' : None}, recursive=False) # Get all proceedings
            for proceeding in proceedings:
                if pdf_count < MAX_PDF_COUNT:
                    papers = proceeding.findChildren("p")[1:PAPERS_PER_PROCEEDING+1]
                    pdf_links = [paper.find('a').get('href') for paper in papers]
                    downloaded = list((map(download_pdf, pdf_links)))
                    valid_pdfs = list(filter(lambda status: status, downloaded))
                    # len(valid_pdfs) != len(pdf_links) and print(valid_pdfs)
                    pdf_count += len(valid_pdfs)






def parse_acl_home_page():
    """
        Parse home page by venue
    """
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
            print(proceeding_link)
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
            
            
        
        
    # tags = tables[0].findAll(lambda tag: tag.name=='tr')
    # tds = list(map(lambda tag: tag.findAll("td"), tags))


parse_acl()
# parse_acl_home_page()