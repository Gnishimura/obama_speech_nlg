# Forked and authored by yimihua2013 (with some edits/customizing)
from pymongo import MongoClient
import requests
import time
import random
import re
from bs4 import BeautifulSoup

"""Use Python's BeautifulSoup to scrape AmericanRhetoric.com for all of Obama's speeches"""

website = "http://www.americanrhetoric.com/barackobamaspeeches.htm"
r = requests.get(website)
page = BeautifulSoup(r.content, features='lxml')

# Part 1: grab all the speech urls

def filter_link(link):
    """Grab the urls from every speech link on the obama speeches page 
    in American Rhetoric"""
    href = link.get('href')
    if href:
        return href.startswith('speeches') and href.endswith('.htm')

links = page.find_all('a')
urls = filter(filter_link, links)
urls = [url.get('href') for url in urls]
print(len(urls))  #467 speeches in total

def get_url(link):
    """Form the full url"""
    pre = 'http://www.americanrhetoric.com/'
    return pre + link

full_urls = []
for url in urls:
    full_urls.append(get_url(url))

# Part 2: grab the speech content and save into MongoDB

# Get the speech content
def get_speech_content(url):
    """Parse the page data to get only the actual speeches"""
    re = requests.get(url)
    time.sleep(5 + random.random() * 10)
    page_data = BeautifulSoup(re.content, features='lxml')
    # Search the page for only content in Verdana font, since that
    # appears to be what all the speech text is written in
    speech_data = page_data.find_all("font", {"face": "Verdana"})
    return speech_data

# Initialize MongoDB to save speeches
obamongo = MongoClient()
obama_db = obamongo['obama_db']
speech_collection = obama_db['speeches']

def scrape_ar(url):
    """Scrape pages at random 1-11 sec intervals"""
    re = requests.get(url)
    page_data = BeautifulSoup(re.content)
    speech_data = page_data.find_all("font", {"face": "Verdana"})
    time.sleep(1 + random.random() * 10)
    return speech_data

def clean_speech(speech_data):
    """Clean up the html of the speeches in a really brute force-y way"""
    a = re.sub('<[^<]+?>', '', str(speech_data))
    b = re.sub('\r\n\t\t', '', str(a))  
    c = re.sub('\.,', '.', str(b))
    d = re.sub('\r\n', '', str(c))
    e = re.sub('\n', '', str(d))
    f = re.sub('\xa0', '', str(e))
    h = re.sub('\t', '', str(f))
    return print(h)

def save_speech(url, speech_data):
    """Save speeches into MongoDB collection"""
    name = url.split('/')[-1].split('.')[0]
    speech_collection.delete_many({'name': name})
    speech_collection.insert_one({'name': name, 'speech': speech_data})
    # print('Saving {}...'.format(name))

def compile_collection(full_urls):
    """Scrape, clean, save"""
    for url in full_urls:
        print(url)
        speech_data = scrape_ar(url)
        speech_text = clean_speech(speech_data)
        save_speech(url, speech_data)
    

compile_collection(full_urls)


