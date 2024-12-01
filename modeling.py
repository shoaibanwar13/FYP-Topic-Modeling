from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List
import re
from collections.abc import Iterable
import matplotlib
import xml.etree.ElementTree as ET
import csv
import gensim
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim import corpora
from wordcloud import WordCloud
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import spacy
nlp = spacy.load("en_core_web_sm")  # Use the appropriate model
from pprint import pprint
import nltk
nltk.download('stopwords')

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from gensim.models.coherencemodel import CoherenceModel
# Visualize the topics
import os
#from settings import PROJECT_ROOT
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
from django.http import JsonResponse
from typing import List, Dict


app = FastAPI()

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function for LDA processing
def generate_topics(data: List[str], num_topics: int = 5, num_words: int = 10):
    vectorizer = CountVectorizer(stop_words="english")
    data_vectorized = vectorizer.fit_transform(data)

    lda_model = LatentDirichletAllocation(
        n_components=num_topics, random_state=42
    )
    lda_model.fit(data_vectorized)

    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda_model.components_):
        topics.append({
            f"Topic {idx + 1}": [words[i] for i in topic.argsort()[-num_words:]]
        })
    return topics

# Endpoint for uploading the CSV file and generating LDA topics

app = FastAPI()

@app.post("/upload/",response_model=List[Dict[str, str]])
async def upload_csv(
    file1: UploadFile,
    file2:UploadFile,
    file3:UploadFile,
    file4:UploadFile,
    column_name: str = Form(...),
):
    try:
        # Load the Excel file
        data = pd.read_excel(file1.file)

        # Check if the specified column exists
        if column_name not in data.columns:
            return JSONResponse(
                content={"error": f"Column '{column_name}' not found in the file."},
                status_code=400
            )

        # Drop specific columns and process the DataFrame
        papers = data.drop(columns=['PostTypeId', 'Score', 'ViewCount'], axis=1, errors='ignore')

        # Ensure 'Tags' exists in the DataFrame
        if 'Tags' in papers.columns:
            # Clean the 'Tags' column
            papers['Tags_processed'] = papers['Tags'].map(lambda x: re.sub(r'[,!?]', '', str(x)))  # Remove punctuation
            papers['Tags_processed'] = papers['Tags_processed'].map(lambda x: re.sub(r'[^a-zA-Z\-\.\s]+', ' ', x))  # Remove non-alphabets except hyphen
            papers['Tags_processed'] = papers['Tags_processed'].map(lambda x: x.lower())  # Convert to lowercase
        else:
            return JSONResponse(
                content={"error": "'Tags' column not found for processing."},
                status_code=400
            )
        tags = papers['Tags_processed']
        
        new_tags = []
        for i in tags:
          new_tags.append(i.split())
       
        def flatten(lis):
            for item in lis:
              if isinstance(item, Iterable) and not isinstance(item, str):
                 for x in flatten(item):
                   yield x
              else:        
                 yield item
        new_tags = list(flatten(new_tags))   # make flate list of the Array new_ tags 
             # making unique list of Tags 
        unique_list = []
        for unique in new_tags:
          if unique not in unique_list:
              unique_list.append(unique)
        len(unique_list) #Find and Print number of Unique Tags  
         
        
#print(unique_list)
        # Convert unique_list to a DataFrame
        uunique_list = pd.DataFrame(unique_list)

# Add <> to each element
        uunique_list = uunique_list.applymap(lambda x: f"<{x}>")

# Write the updated DataFrame to an Excel file
        uunique_list.to_excel('unique_tags.xlsx', index=False, header=True)

# Read back the Excel file
        read_uunique_list = pd.read_excel("unique_tags.xlsx")
       

# Convert to a list
        read_uunique_list_l = read_uunique_list.values.tolist()

# Flatten the list of lists and print the updated list
        nnlist = [j for sublist in read_uunique_list_l for j in sublist]
        print(nnlist)
            
        pt = pd.read_excel(file2.file)        #tags in <> brackets 
        data2=pt
        df=pd.DataFrame(data2)   # make data frame of whole dataset file D
        (df['Tags'] =='flask').sum()
        strr = 'rest'     #makingi token to search no of question with each tag in Data set D
        str_with_tag = '<'+strr+'>' 
        print(str_with_tag)   
        count = 0
        for s in range(len(pt['Tags'])):   #count the rest Tags posts in dataset D
           if 'rest' in pt['Tags'][s]:
               count=count+1
        print(count)
        utags = pd.read_excel(r'unique_tags.xlsx') # reading the file of Unique tags which created from data p
        utags = utags[0]
        # TOTAL NUMBER OF UNIQUE TAGS
        print(print(utags))
    #     D = []     # Number of questions posts with each unique tag in dataset D

    #     for ut in range(len(utags)):
    #         count = 0
    #         if pd.isnull(utags[ut]):
    #            continue
    #         for tg in range(len(pt['Tags'])):
    # #for tg in range(60428):
    #     #str_with_tag = '<'+utags[ut]+'>'
    #     #if str_with_tag in pt['Tags'][tg]:
    #            if utags[ut] in pt['Tags'][tg]:
    #                count=count+1
    #         D.append(count)
    #     DD = pd.DataFrame(D)
    #     print(DD)
         #DD.to_excel('DD.xlsx',index=False,header=True) 

         
        df_1=pd.DataFrame(papers)
        df_2=df_1.Tags_processed.unique()
        print(df_2)
        ppp = papers['Tags']                #link of Dataset file P(WEB1.xlsx)
        P = [] 
        for ut in range(len(utags)):
            count = 0
            if pd.isnull(utags[ut]):
               continue
            for tg in range(len(papers['Tags'])):
        #str_with_tag = '<'+utags[ut]+'>'
                if utags[ut] in papers['Tags'][tg]:
                    count=count+1
            P.append(count)
        PP = pd.DataFrame(P)
        PP.to_excel('PP.xlsx',index=False,header=True) 
        a2 = pd.read_excel('DD.xlsx')
        a3 = pd.read_excel('PP.xlsx')
        a1 = pd.read_excel('unique_tags.xlsx')
        a1 = a1.rename(columns = {0:"unique_tags"})
        a2 = a2.rename(columns = {0:"D"})
        a3 = a3.rename(columns = {0:"P"})
    #print(str_with_tag + ' = ' + str(count))
        frame = [a1,a2,a3]
        new_list_of_tags = pd.concat(frame,axis=1)
        print(new_list_of_tags)
        length=len(papers['Tags'])
        new_list_of_tags["Significance"] = new_list_of_tags["P"]/new_list_of_tags["D"]
        new_list_of_tags["Relevance"] = new_list_of_tags["P"]/length
        print(new_list_of_tags)
        new_list_of_tags.to_csv("final_unique_tags.csv",index=False,header=True)
        data3 = pd.read_csv("final_unique_tags.csv")
        sig = data3["Significance"]
        sig = sig.to_list()
        new_sig = []
        for i in sig:
           new_sig.append(round(i, 3))
        new_sig[0]
        rel = data3["Relevance"]
        rel = rel.to_list()
        new_rel = []
        for i in rel:
           new_rel.append(round(i, 4))
        new_rel[0]
        # for the no of total recommned Tags with Threshoulds
        value11 = [0.05,0.1,0.15,0.2,0.25,0.3,0.35]
        value12 = [0.001,0.005,0.001,0.002,0.025,0.03]
        value2 = 0.001
        
        for i in range(len(value11)):
            value1=value11[i]
            count=0
            for j in range(len(value12)):
                value2=value12[j]
                for tg in range(len(new_rel)):       
                    if new_sig[tg]>=float(value1) and new_rel[tg]>= value2:
                       count=count+1
                print("signifance=",value1, "relevance=", value2, "count=",count)
                df=pd.DataFrame(columns=["value1","value2","count"])
        results=[]
        value11 = [0.05,0.1,0.15,0.2,0.25,0.3,0.35]
        value12 = [0.001,0.005,0.001,0.002,0.025,0.03]
        df = pd.DataFrame(columns=["signifance","relevance","count"])
        for i in range(len(value11)):
            value1=value11[i]
            max1=0
            s=0
            r=0
            for j in range(len(value12)):
        #max1=0
               count=0
               value2=value12[j]
               for tg in range(len(new_rel)):       
                   if new_sig[tg]>=float(value1) and new_rel[tg]>= value2:
                       count=count+1
            if max1 < count:
                    max1=count
                    s= value1
                    r= value2
            results.append({'signifance': s, 'relevance': r, 'count': max1})
            print("=====================================================")
            print("signifance=",s, "relevance=", r, "max_count=",max1,)
        uniqueposts = pd.DataFrame(results)
        uniqueposts.to_excel('UniquePosts.xlsx',index=False,header=True)    
#         Total_rec =[28,30,34,42,57,90,162]
# #print(signifance)
#         Total_revlnet = [28,29,30,36,46,52,110]
#         index = ['U= 0.35 v= 0.001', 'u= 0.3 v= 0.001', 'u= 0.2 v= 0.001','u= 0.25 v= 0.001',
#          'u= 0.1 v= 0.001', 'u= 0.15 v= 0.001', 'u= 0.05 v= 0.001']
#         df6 = pd.DataFrame({'Total recomendened': Total_rec,'Total relevant': Total_revlnet,}, index=index)
#         ax = df6.plot.barh()
#         s_Tags = pd.read_excel(file3.file)  # prepared by discussion
#         print(s_Tags)
#         T=unique_list



#         FILE = file4.file
# #TAGS = ['api', 'rest','xml','facebook','wcf','soap','web-services']  will be used to take specich tag dfataset  
#         TAGS = T

#         COLS = ["Id", "PostTypeId", "AcceptedAnswerId", "ParentId", "CreationDate", "DeletionDate", "Score", "ViewCount", "Body", "OwnerUserId", "OwnerDisplayName", "LastEditorUserId", "LastEditorDisplayName", "LastEditDate", "LastActivityDate", "Title", "Tags", "AnswerCount", "CommentCount", "FavoriteCount", "ClosedDate", "CommunityOwnedDate", "ContentLicense"]
 
#         context = ET.iterparse(FILE, events=("start", "end"),
#                        parser=ET.XMLParser(encoding='utf-8'))

#         print("Going to extract questions")
#         with open('output_1.csv', 'w', newline='', encoding='utf-8') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow(COLS)
#             _, root = next(context)
#             for event, elem in context:
#                 if event == "end" and elem.tag == "row":
#                    tags = elem.attrib.get('Tags', 'None')
#                    for tag in TAGS:
#                        if tag in tags:
#                           data = []
#                           for col in COLS:
#                               data.append(elem.attrib.get(col, ''))
#                               csvwriter.writerow(data)
#                               continue
#                    if int(elem.attrib['Id']) % 100000 == 0:
#                        print('done', elem.attrib['Id'])
#                    elem.clear()
#                    root.clear()
#         print("Extraction Over")
        remove_duplicate = pd.read_csv(file4.file)
        new_file = remove_duplicate.drop_duplicates(subset=['Id'])
        print(new_file.head())
        new_file.to_csv("output_2.csv",index=False, header=True)
        papers = pd.read_csv("output_2.csv")  #  dataset of all unique tags  
# Remove punctuation
        papers = papers.replace('\n',' ', regex=True)   #replacew newline with space    
        papers['Title_text_processed'] = papers['Title'].map(lambda x: re.sub('[,\.!?]', ' ', x)) #remove punct!
        papers['Title_text_processed'] = papers['Title_text_processed'].map(lambda x: re.sub('<[^>]+>', ' ', x)) #remove html tag
        papers['Title_text_processed'] = papers['Title_text_processed'].map(lambda x: re.sub('[^a-zA-Z ]+', ' ', x)) #remove non alpahbetics  tag
        # Convert the titles to lowercase
        papers['Title_text_processed'] = papers['Title_text_processed'].map(lambda x: x.lower())

# Print out the first rows of papers
        print(papers.head())
        #Saving File  with distant posts 
        papers.to_csv('processed.csv',index=False,header=True)
        print(len(papers))
        def sent_to_words(sentences):     
            for sentence in sentences:
                 yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

        data = papers['Title_text_processed'].values.tolist()
        data_words = list(sent_to_words(data))
        print(data_words[:1][0][:30])
        # Join the different processed titles together.
        long_string = ','.join(list(papers['Title_text_processed'].values))

# Create a WordCloud object
        wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
        wordcloud.generate(long_string)

# Visualize the word cloud
        wordcloud.to_image()
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
        print(trigram_mod[bigram_mod[data_words[0]]])
        def remove_stopwords(texts):
          return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
                """https://spacy.io/api/annotation"""
                texts_out = []
                for sent in texts:
                    doc = nlp(" ".join(sent)) 
                    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                return texts_out
        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        print(data_lemmatized[:1])
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # View
        print(corpus[:1])
        # Human readable format of corpus (term-frequency)
        [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

#         lda_model = gensim.models.LdaMulticore(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=num_topics, 
#                                            random_state=100,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha=0.01,
#                                            eta=0.91)
        

# # Print the Keyword in the 10 topics
#         pprint(lda_model.print_topics())
#         doc_lda = lda_model[corpus]
        # save Traing LADA model to CSV file 
        # PP = pd.DataFrame(lda_model.print_topics(-1,num_words=20))
        # PP.to_csv('Gensim_LDA1.csv',index=False,header=True)
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                            num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        # Print the Keyword in the 20 topics
        pprint(lda_model.print_topics(-1,num_words=20))
        doc_lda = lda_model[corpus]
        # save Traing LADA model to CSV file 
         
        PP = pd.DataFrame(lda_model.print_topics(-1,num_words=20))
         
        PP.to_csv('Gensim_LDA1.csv',index=False,header=True)
        # Compute Perplexity (inability to deal with or understand something)
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
       
# Visualize the topics
        # pyLDAvis.enable_notebook()
        # pyLDAvis.enable_notebook()
        # vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

        # pyLDAvis.save_html(vis, 'ldavis_tuned'+ str(num_topics) +'.html')
       
        # Prepare the response
        response = {
            "Topics": [{"index": index, "content": content} for index, content in lda_model],
            "Coherence Score":coherence_lda,
             
            "Perplexity":lda_model.log_perplexity(corpus)

        }

        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
