#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:21:51 2018

@author: Chris
"""

import json
import os

import requests
from requests.exceptions import HTTPError

BASE_URL = 'https://api.figshare.com/v2/{endpoint}'
TOKEN = 'b3111b15eff0c014f352360e9019567ab703712552287127cef51de5d56117d7e9453ad7f5925b5ded81982b7bf0eba3ffadd64c297f3397a7db4124adec9735'
CHUNK_SIZE = 1048576

class DataRetriever(object):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def download_collection_files(self, collection_id):
        self.check_data_dir()
        articles = get_articles_from_collection(collection_id)
        download_urls = []
        for article in articles:
            download_urls = download_urls + get_download_urls(article)
        self.download_files(download_urls)
    
    def check_data_dir(self):
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
    
    def download_files(self, download_urls):
        for download_url in download_urls:
            if self.already_downloaded(download_url):
                continue
            self.download_file(download_url)
    
    def already_downloaded(self, download_url):
        filename = download_url.split('/')[-1]
        local_path = "{:s}/{:s}".format(self.data_dir, filename)
        if os.path.exists(local_path):
            "Skipping {:s}".format(filename)
            return True
        return False
    
    def download_file(self, download_url):
        response = requests.request('GET', download_url)
        if response.status_code == 200:
            filename = download_url.split('/')[-1]
            local_path = "{:s}/{:s}".format(self.data_dir, filename)
            print "Downloading {:s}".format(filename)
            with open(local_path, 'wb') as f:
                for chunk in response:
                    f.write(chunk)

def raw_issue_request(method, url, data=None, binary=False):
    headers = {'Authorization': 'token ' + TOKEN}
    if data is not None and not binary:
        data = json.dumps(data)
    response = requests.request(method, url, headers=headers, data=data)
    try:
        response.raise_for_status()
        try:
            data = json.loads(response.content)
        except ValueError:
            data = response.content
    except HTTPError as error:
        print 'Caught an HTTPError: {}'.format(error.message)
        print 'Body:\n', response.content
        raise

    return data

def issue_request(method, endpoint, *args, **kwargs):
    return raw_issue_request(method, BASE_URL.format(endpoint=endpoint), *args, **kwargs)
    
def get_articles_from_collection(id):
    articles = issue_request('GET', "collections/{:d}/articles".format(id))
    return [article['id'] for article in articles]

def get_download_urls(id):
    article_files = issue_request('GET', "articles/{:d}/files".format(id))
    return [article_file['download_url'] for article_file in article_files]
#%%
    
