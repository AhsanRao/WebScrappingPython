import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
import time
from nltk.util import pr
import numpy
import pandas
import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import csv
from csv import DictWriter
from csv import writer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


def getVerbs(text):
    tok = []
    tok = word_tokenize(text)
    temp = pos_tag(tok)
    verb = []
    length = len(temp)
    for i in range(length):
        if temp[i][1] == 'VB' or temp[i][1] == 'VBD' or temp[i][1] == 'VBG' or temp[i][1] == 'VBN' or temp[i][1] == 'VBP' or temp[i][1] == 'VBZ':
            verb.append(temp[i][0])
    return verb


def getNouns(text):
    tok = []
    tok = word_tokenize(text)
    temp = pos_tag(tok)
    noun = []
    length = len(temp)
    for i in range(length):
        if temp[i][1] == 'NN' or temp[i][1] == 'NNS' or temp[i][1] == 'NNP' or temp[i][1] == 'NNPS':
            noun.append(temp[i][0])
    return noun


def getAdjective(text):
    tok = []
    tok = word_tokenize(text)
    temp = pos_tag(tok)
    adjective = []
    length = len(temp)
    for i in range(length):
        if temp[i][1] == 'JJ' or temp[i][1] == 'JJR' or temp[i][1] == 'JJS':
            adjective.append(temp[i][0])
    return adjective


def qualityNoun(noun):
    for i in range(len(noun)):
        if noun[i] == "quality" or noun[i] == "Quality":
            for x in range(5):
                print(noun[i-x], ",")


# Website 1 : NUCES FAST
options = Options()
scrape = webdriver.Chrome(ChromeDriverManager().install(), options=options)
text = []
scrape.get('http://nu.edu.pk/Home')
time.sleep(5)
text = scrape.find_element_by_class_name('col-md-12').text
scrape.get('http://nu.edu.pk/vision-and-mission')
time.sleep(5)
text += scrape.find_element_by_class_name('col-md-12').text
scrape.get('http://nu.edu.pk/QEC')
time.sleep(5)
text += scrape.find_element_by_class_name('col-md-6').text
scrape.get('https://www.nu.edu.pk/Program/MS(SE)')
time.sleep(5)
text += scrape.find_element_by_class_name('col-md-12').text
scrape.get('https://www.nu.edu.pk/QEC/BuildingCalendar')
time.sleep(5)
text += scrape.find_element_by_class_name('col-md-12').text

print("\nScraped Text: ", text)
print("\n Verbs Of Website 1")
print(getVerbs(text))
print("\n Nouns Of Website 1")
print(getNouns(text))
print("\n Adjective Of Website 1")
print(getAdjective(text))
verb1 = getVerbs(text)
adjective1 = getAdjective(text)
noun1 = getNouns(text)

# Website 2: LUMS
scrape.get('https://lums.edu.pk/news/students-attend-tkxel-campus-drive-lums')
time.sleep(5)
text = scrape.find_element_by_class_name('body-custom').text
scrape.get('https://lums.edu.pk/news/soe-models-educational-innovation-aga-khan-education-service-pakistan')
time.sleep(5)
text += scrape.find_element_by_class_name('body-custom').text
scrape.get('https://lums.edu.pk/news/international-professors-conduct-teaching-and-learning-workshop-further-lums-no-borders-agenda')
time.sleep(5)
text += scrape.find_element_by_class_name('body-custom').text
scrape.get(
    'https://lums.edu.pk/news/lums-annual-dinner-recognises-universitys-generous-supporters')
time.sleep(5)
text += scrape.find_element_by_class_name('body-custom').text
scrape.get('https://lums.edu.pk/news/department-chemistry-chemical-engineering-students-visit-nestle-pakistan')
time.sleep(5)
text += scrape.find_element_by_class_name('body-custom').text

print('\n Scraped Text: ')
print(text)
print("\n Verbs Of Website 2")
print(getVerbs(text))
print("\n Nouns Of Website 2")
print(getNouns(text))
print("\n Adjective Of Website 2")
print(getAdjective(text))
verb2 = getVerbs(text)
adjective2 = getAdjective(text)
noun2 = getNouns(text)

# Website 3: FCC
scrape.get('https://www.fccollege.edu.pk/mphil-food-safety-and-quality-management/')
time.sleep(5)
text = scrape.find_element_by_class_name('tab-content').text
scrape.get('https://www.fccollege.edu.pk/mphil-food-safety-and-quality-management/')
time.sleep(5)
text += scrape.find_element_by_class_name('tab-content').text
scrape.get(
    'https://www.fccollege.edu.pk/department-of-computer-science/collaborations/')
time.sleep(5)
text += scrape.find_element_by_class_name('tab-content').text
scrape.get(
    'https://www.fccollege.edu.pk/forman-journal-of-business-and-innovation/')
time.sleep(5)
text += scrape.find_element_by_class_name('tab-content').text
scrape.get(
    'https://www.fccollege.edu.pk/important-notice-for-beginning-of-fall-semester/')
time.sleep(5)
text += scrape.find_element_by_class_name('fusion-text.fusion-text-1').text

print('\n Scraped Text: ')
print(text)
print("\n", text)
print("\n Verbs Of Website 3")
print(getVerbs(text))
print("\n Nouns Of Website 3")
print(getNouns(text))
print("\n Adjective Of Website 3")
print(getAdjective(text))
verb3 = getVerbs(text)
adjective3 = getAdjective(text)
noun3 = getNouns(text)

# Bar Chart
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

Website1 = [len(verb1), len(adjective1), len(noun1)]
Website2 = [len(verb2), len(adjective2), len(noun2)]
Website3 = [len(verb3), len(adjective3), len(noun3)]

br1 = np.arange(len(Website1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

plt.bar(br1, Website1, color='r', width=barWidth,
        edgecolor='grey', label='Website1')
plt.bar(br2, Website2, color='g', width=barWidth,
        edgecolor='grey', label='Website2')
plt.bar(br3, Website3, color='b', width=barWidth,
        edgecolor='grey', label='Website3')

plt.xlabel('Word Type', fontweight='bold', fontsize=15)
plt.ylabel('No of Words', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(Website1))],
           ['Verb', 'Adjective', 'Noun'])

plt.legend()
plt.show()

# Visualization
data_list = [verb1, adjective1, noun1, verb2,
             adjective2, noun2, verb3, adjective3, noun3]
temp = []
for i in data_list:
    temp.append(len(i))
max_length = max(temp)
for i in data_list:
    while len(i) < max_length:
        i.append(None)
temp = np.array(data_list)

df = pandas.DataFrame(
    data={"Verbs 1": temp[0], "Nouns 1": temp[1], "Adjective 1": temp[2], "Verbs 2": temp[3], "Nouns 2": temp[4], "Adjective 2": temp[5], "Verbs 3": temp[6], "Nouns 3": temp[7], "Adjective 3": temp[8]})
df.to_csv("./data.csv", sep=',', index=False)
pd.set_option('display.max_columns', None)

write = pd.read_csv('data.csv', names=[
    '   ', 'Website 1', '----', ' ', 'Website 2', ' --', '-- ', 'Website 3', ''])
# write.head(len(temp))
print(write)

# Top 10 Nouns
noun1_count = Counter(noun1)
noun1_common = noun1_count.most_common()
noun2_count = Counter(noun2)
noun2_common = noun2_count.most_common()
noun3_count = Counter(noun3)
noun3_common = noun3_count.most_common()
print("\nTop 10 Nouns of Website 1\n")
for x in range(1):
    print(noun1_common[x])
print("\nTop 10 Nouns of Website 2\n")
for x in range(1):
    print(noun2_common[x])
print("\nTop 10 Nouns of Website 3\n")
for x in range(1):
    print(noun3_common[x])

# Quality Adjective (part c)
print("\nAdjective Adjecent of 'Quality' Adjective")
print("\nWebsite 1: ")
qualityNoun(adjective1)
print("\nWebsite 2: ")
qualityNoun(adjective2)
print("\nWebsite 3: ")
qualityNoun(adjective3)
