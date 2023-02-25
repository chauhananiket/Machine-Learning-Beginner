import os
import numpy as np
import shutil
import re
import math
import numpy
import pandas as pd
import csv
import math
import string
import sys
import fileinput
import json
import urllib
import urllib3
import requests
import zipfile
import time
import argparse
import pickle
from termcolor import colored, cprint
import colorama
import webbrowser
import base64
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import sklearn.ensemble as ske
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

class Entropy():
    def __init__(self, data):
        self.data = data
    def range_bytes(): return range(256)
    def range_printable(): return (ord(c) for c in string.printable)
    def H(self, data, iterator=range_bytes):
        if not data:
            return 0
        entropy = 0
        for x in iterator():
            p_x = float(data.count(chr(x)))/len(data)
            if p_x > 0:
                entropy += - p_x*math.log(p_x, 2)
        return entropy

class URLFeatures():
    def bag_of_words(self, url):
        vectorizer = CountVectorizer()
        content = re.split('\W+', url)
        X = vectorizer.fit_transform(content)
        num_sample, num_features = X.shape
        return num_features
    def contains_IP(self, url):
        check = url.split('/')
        reg = re.compile("^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$")
        result = 0
        for item in check:
            if re.search(reg, item):
                result = 1
        return result
    def url_length(self, url):
        return len(url)
    def special_chars(self, url):
        counter = 0
        if '*' in url:
            counter += 1
        if ';' in url:
            counter += 1
        if '%' in url:
            counter += 1
        if '!' in url:
            counter += 1
        if '&' in url:
            counter += 1
        if ':' in url:
            counter += 1
        return counter

    def suspicious_strings(self, url):
        counter = 0
        if '.exe' in url:
            counter += 1
        if 'base64' in url:
            counter += 1
        if '/../' in url:
            counter += 1
        if '.pdf' in url:
            counter += 1
        if 'free' in url:
            counter += 1
        if 'Free' in url:
            counter += 1
        if 'FREE' in url:
            counter += 1
        if '.onion' in url:
            counter += 1
        if '.tor' in url:
            counter += 1
        if '.top' in url:
            counter += 1
        if '.bid' in url:
            counter += 1
        if '.ml' in url:
            counter += 1
        if 'bitcoin' in url:
            counter += 1
        if '.bit' in url:
            counter += 1
        if '.php?email=' in url:
            counter += 1
        if 'cmd=' in url:
            counter += 1
        return counter

    def num_digits(self, url):
        numbers = sum(i.isdigit() for i in url)
        return numbers
    def popularity(self, url):
        result = 0
        domain = url.split('/', 1)[-1]
        with open(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\top1m_rank.csv', 'rt') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if domain == row[1]:
                    result = row[0]            
        return int(result)

class RetrieveData():
    def __init__(self):
        self.openphish = 'https://openphish.com/feed.txt'
        self.umbrella = 'http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip'
    def get_malicious_urls(self):
        open_phish = self.openphish
        urllib.request.urlretrieve(open_phish, r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\openphish.txt')
        df = pd.read_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\openphish.txt', sep='\n', header=None)
        df= df.iloc[0:3000]
        df.columns = ['URL']
        df.to_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\malicious_urls.csv', sep='\n', index=False)
    def get_benign_urls(self):  
        try:
            umbrella = self.umbrella
            urllib.request.urlretrieve(umbrella, r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\top1m.csv.zip')
            print("[+] Unzipping Benign URL data...\n")
            with zipfile.ZipFile(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\top1m.csv.zip') as zip_ref:
                zip_ref.extractall(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\\')
            df = pd.read_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\top1m.csv.zip', header=None)
            df.columns = ['Rank', 'URL']
            df.to_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\top1m_rank.csv', sep='|', index=False)
            df = df.drop('Rank', axis=1)
            df = df.iloc[0:2000]
            df.to_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\top.csv', sep='|', index=False)
            df.to_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\benign_urls.csv', sep='|', index=False)
        except:
            print("[-] Unable to retrieve an updated URL list.\n")

class SafeBrowse():
    def __init__(self, apikey):
        self.safe_base = 'https://safebrowsing.googleapis.com/v4/threatMatches:find?key=%s' % (apikey)
        self.platform_types = ['ANY_PLATFORM']
        self.threat_types = ['THREAT_TYPE_UNSPECIFIED',
                             'MALWARE', 
                             'SOCIAL_ENGINEERING', 
                             'UNWANTED_SOFTWARE', 
                             'POTENTIALLY_HARMFUL_APPLICATION']
        self.threat_entry_types = ['URL']
    def set_threat_types(self, threats):
        self.threat_types = threats
    def set_platform_types(self, platforms): 
        self.platform_types = platforms
    def threat_matches_find(self, *urls): 
        try:
            threat_entries = []
            results = {}
            for url_ in urls: 
                url = {'url': url_} 
                threat_entries.append(url)
            request_body = {
                'client': {
                    'clientId': 'MLURL_CLIENT',
                    'clientVersion': '1.0'
                },
                'threatInfo': {
                    'threatTypes': self.threat_types,
                    'platformTypes': self.platform_types,
                    'threatEntryTypes': self.threat_entry_types,
                    'threatEntries': threat_entries
                }
            }
            headers = {'Content-Type': 'application/json'}
            r = requests.post(self.safe_base, 
                            data=json.dumps(request_body), 
                            headers=headers, timeout=2)
            jdata = r.json()
            if jdata['matches'][0]['threatEntryType'] == 'URL':
                return 1
            else:
                return 0
        except:
            return 0

def parse_url(url):
    if 'http://' in url:
        url_http = url.split('http://', 1)[-1]
        return url_http
    elif 'https://' in url:
        url_https = url.split('https://', 1)[-1]
        return url_https
    else:
        return url
  
def get_url_info(url):
    features = {}
    parsed_url = parse_url(url)
    getEntropy = Entropy(parsed_url)
    entropy = getEntropy.H(parsed_url)
    features['Entropy'] = entropy
    feature = URLFeatures()
    features['BagOfWords'] = feature.bag_of_words(parsed_url)
    features['ContainsIP'] = feature.contains_IP(parsed_url)
    features['LengthURL'] = feature.url_length(parsed_url)
    features['SpecialChars'] = feature.special_chars(parsed_url)
    features['SuspiciousStrings'] = feature.suspicious_strings(url)
    features['NumberOfDigits'] = feature.num_digits(parsed_url)
    features['Popularity'] = feature.popularity(parsed_url)
    apikey = base64.b64decode('QUl6YVN5QV9XbU53MHRyZTEybWtMOE1qYUExY0c3Smd4SnRuU0lv')
    apikey = apikey.decode('utf-8')
    safe = SafeBrowse(apikey)
    features['Safebrowsing'] = safe.threat_matches_find(url) 
    return features
      
def extract_features(url):
    features = []
    parsed_url = parse_url(url)
    features.append(url)
    getEntropy = Entropy(parsed_url)
    entropy = getEntropy.H(parsed_url)
    features.append(entropy)
    feature = URLFeatures()
    features.append(feature.bag_of_words(parsed_url))
    features.append(feature.contains_IP(parsed_url))
    features.append(feature.url_length(parsed_url))
    features.append(feature.special_chars(parsed_url))
    features.append(feature.suspicious_strings(url))
    features.append(feature.num_digits(parsed_url))
    features.append(feature.popularity(parsed_url))
    apikey = base64.b64decode('QUl6YVN5Qzl0c3gzcFlmQXhPN25PSGE5UWtNdjR6VW1QNk90UmQw')
    apikey = apikey.decode('utf-8')
    safe = SafeBrowse(apikey)
    response = safe.threat_matches_find(url) 
    features.append(response)
    return features

def create_dataset():
    output_file = r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\data_urls.csv'
    csv_delimeter = '|'
    csv_columns = [
        "URL",
        "Entropy",
        "BagOfWords",
        "ContainsIP",
        "LengthURL",
        "SpecialChars",
        "SuspiciousStrings",
        "NumberOfDigits",
        "Popularity",
        "Safebrowsing",
        "Malicious", 
    ]
    feature_file = open(output_file, 'a')
    feature_file.write(csv_delimeter.join(csv_columns) + "\n")
    with open(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\malicious_urls.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter='\n')
        rows = list(reader)
    for row in rows:
        print('\n[+] Extracting features from ', row['URL'])
        try:
            e = extract_features(row['URL'])
            e.append(1)
            feature_file.write(csv_delimeter.join(map(lambda x: str(x), e)) + "\n")
            print(colored('\n[*] ', 'green') + "Features extracted successfully.\n")
        except:
            print("[-] Error: Unable to extract features.\n")
    with open(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\benign_urls.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        rows = list(reader)

    for row in rows:
        print('\n[+] Extracting features from ', row['URL'])
        try:
            e = extract_features(row['URL'])
            e.append(0)
            feature_file.write(csv_delimeter.join(map(lambda x: str(x), e)) + "\n")
            print(colored('\n[*] ', 'green') + "Features extracted successfully.\n")
        except:
            print("[-] Error: Unable to extract features.\n")
    feature_file.close()
    
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def train_model():
    df = pd.read_csv(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\data_urls.csv', sep='|')
    X = df.drop(['URL', 'Malicious'], axis=1).values
    y = df['Malicious'].values
    print("PRINTING DATASET")
    print(df.head())
    print(df.describe())
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42) 
    print("\t[*] Training samples: ", len(X_train))
    print("\t[*] Testing samples: ", len(X_test))
      # KNN
    k=9
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predict_knn= knn.predict(X_test)
    score_knn=accuracy_score(y_test, y_predict_knn)
    print('Accuracy of KNN',score_knn) 
    all_features = X.shape[1]
    features = []
    for feature in range(all_features):
        features.append(df.columns[1+feature])
    try:
        print("\n[+] Saving algorithm and feature list in classifier directory...")
        joblib.dump(knn, r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\Classifier\classifier.pkl')
        open(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\Classifier\features.pkl', 'wb').write(pickle.dumps(features))
        print(colored('\n[*] ', 'green') + " Saved.")
    except:
        print('\n[-] Error: Algorithm and feature list not saved correctly.\n')

def classify_url(url):
    clf = joblib.load(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\Classifier\classifier.pkl'))
    features = pickle.loads(open(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\Classifier\features.pkl'),
        'rb').read())
    data = get_url_info(url)
    feature_list = list(map(lambda x:data[x], features))
    print('Features of Url passed \n',feature_list)
    result = clf.predict([feature_list])[0]

    if result == 0:
        print(colored('\n[*] ', 'green') + "MLURL has classified URL %s as " % url + colored("benign", 'green') + '.')
    else: 
        print(colored('\n[*] ', 'green') + "MLURL has classified URL %s as " % url + colored("malicious", 'red') + '.')
    return result

def check_valid_url(url):
    print("\n[+] Validating URL format...")
    reg = re.compile('^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$')
    
    if re.match(reg, str(url)):
        print(colored("\n[*] ", 'green') + "URL is valid.")
    else:
        print(colored("\n[-] ", 'red') + "Error: URL is not valid. Please input a valid URL format. ")
        sys.exit(0)

def main():    
    try: 
        getData = RetrieveData()
        getData.get_benign_urls()
        print(colored('\n[*] ', 'green') + "Benign URLs successfully downloaded.\n")
    except:
        print(colored('[-] ', 'red') + "Error: Benign URL downloaded unsuccessful. \n")
    try:
        getData = RetrieveData()
        getData.get_malicious_urls()
        print(colored('\n[*] ', 'green') + "Malicious URLs successfully downloaded.\n")
    except:
        print(colored('\n[-] ', 'red') + "Error: Malicious URL downloaded unsuccessful.\n")

    print("\n[+] Generating URL data...")
    try:
        print("\n[+] Beginning feature extraction...")
        if os.path.exists(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\data_urls.csv'):
            os.remove(r'F:\Python Programs(Machine Learning)\Dataset\Unzipped\Malicious_URL_Detection\data_urls.csv')
            create_dataset()
        else:
            create_dataset()
            (colored("\n[*] ", 'green') + "Feature extraction successful.\n")
    except:
        print(colored("\n[-] ", 'red') + "Error: Feature extraction unsuccessful.\n")
    
    print('\n[+] Training Classification model...\n')
    try:    
        train_model()
        print(colored("\n[*] ", 'green') + "Model successfully trained.")
    except:
        print(colored("\n[-] ", 'red') + "Error: Model unsuccessfully trained .")
    
    #print('\n[+] Running Classifier...')
    #check_valid_url(args.classify)
    #result = classify_url(args.classify)
    #virus_total(result, args.classify)
        
if __name__ == '__main__':
    main()