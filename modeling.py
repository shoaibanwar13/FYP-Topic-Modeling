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
import datetime as dt
from collections import Counter
from scipy.stats import entropy
import nltk
nltk.download('stopwords')
import io 
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
    file4:UploadFile,
    num_topics:int = Form(...),
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
        papers = data.drop(columns=['PostTypeId','DeletionDate','CommunityOwnedDate','ContentLicense'], axis=1, errors='ignore')
        print(papers)

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
        print(len(unique_list))#Find and Print number of Unique Tags  
        
        
         
        
#print(unique_list)
        # Convert unique_list to a DataFrame
        uunique_list = pd.DataFrame(unique_list)
        unique_list_of_tags=uunique_list
        unique_list_of_tags_file=unique_list_of_tags.fillna(0) 
        

# Add <> to each element
        uunique_list.applymap(lambda x: f"<{x}>")

# Write the updated DataFrame to an Excel file
        uunique_list.to_excel('unique_tags.xlsx', index=False, header=True)
        
          

# Read back the Excel file
        read_uunique_list = pd.read_excel("unique_tags.xlsx")
       

# Convert to a list
        read_uunique_list_l = read_uunique_list.values.tolist()
        print(read_uunique_list)

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
        unique_tags_file=utags
        utags = utags[0]
        # # TOTAL NUMBER OF UNIQUE TAGS
        print(utags)
        D = [
    sum(utag in tags for tags in pt['Tags']) 
    for utag in utags 
    if not pd.isnull(utag)
]

     
        df_1=pd.DataFrame(papers)
        df_2 =df_1.Tags_processed.unique()
        print(df_2)
               #link of Dataset file P(WEB1.xlsx)
        P = [
    sum(utag in tags for tags in papers['Tags']) 
    for utag in utags 
    if not pd.isnull(utag)
    ]       
        PP = pd.DataFrame(P)
        PP.to_excel('PP.xlsx',index=False,header=True) 
        a2 = pd.DataFrame(D, columns=["D"])
        a3 = pd.DataFrame(P, columns=["P"])

        a1 = pd.DataFrame(uunique_list)
        a1 = a1.rename(columns = {0:"unique_tags"})
        a2 = a2.rename(columns = {0:"D"})
        a3 = a3.rename(columns = {0:"P"})
        print(str_with_tag + ' = ' + str(count))
        frame = [a1,a2,a3]
        new_list_of_tags = pd.concat(frame,axis=1)
        print(new_list_of_tags)
        length=len(papers['Tags'])
        new_list_of_tags["Significance"] = new_list_of_tags["P"]/new_list_of_tags["D"]
        new_list_of_tags["Relevance"] = new_list_of_tags["P"]/length
        print(new_list_of_tags)
        new_list_of_tags.to_csv("final_unique_tags.csv",index=False,header=True)
        new_list_of_tags_with_s_and_r=new_list_of_tags
        new_list_of_tags_with_s_and_r_file=new_list_of_tags_with_s_and_r.fillna(0)
        
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
        uniqueposts_tags_data=uniqueposts.fillna(0)
       
        uniqueposts.to_excel('UniquePosts.xlsx',index=False,header=True)    
        Total_rec =[28,30,34,42,57,90,162]
    
        Total_revlnet = [28,29,30,36,46,52,110]
        index = ['U= 0.35 v= 0.001', 'u= 0.3 v= 0.001', 'u= 0.2 v= 0.001','u= 0.25 v= 0.001',
         'u= 0.1 v= 0.001', 'u= 0.15 v= 0.001', 'u= 0.05 v= 0.001']
        df6 = pd.DataFrame({'Total recomendened': Total_rec,'Total relevant': Total_revlnet,}, index=index)
         
        # ax = df6.plot.barh()
        #s_Tags = pd.read_excel(file3.file)  # prepared by discussion
        

        FILE = file4.file  # Ensure this is a valid file path or object
        T =['api', 'rest', 'xml', 'facebook', 'wcf', 'soap', 'web-services']

        COLS = ["Id", "PostTypeId", "AcceptedAnswerId", "ParentId", "CreationDate", "DeletionDate", "Score", "ViewCount", 
                "Body", "OwnerUserId", "OwnerDisplayName", "LastEditorUserId", "LastEditorDisplayName", "LastEditDate", 
                "LastActivityDate", "Title", "Tags", "AnswerCount", "CommentCount", "FavoriteCount", "ClosedDate", 
                "CommunityOwnedDate", "ContentLicense"]

        print("Going to extract questions")

        output_csv = io.StringIO()
        csvwriter = csv.writer(output_csv)
        csvwriter.writerow(COLS)

        # Memory-efficient XML parsing
        context = ET.iterparse(FILE, events=("start",), parser=ET.XMLParser(encoding='utf-8'))

        for _, elem in context:
            if elem.tag == "row":
                tags = elem.attrib.get('Tags', '')

                # Check if any tag in T is present in the 'Tags' attribute
                if any(f"<{tag}>" in tags for tag in T):
                    row = [elem.attrib.get(col, '') for col in COLS]
                    csvwriter.writerow(row)

                # Clear the element to free memory
                elem.clear()

        print("Extraction Over")

        # Move cursor back to the beginning for reading
        output_csv.seek(0)

        # Store CSV data in a variable
        csv_data = output_csv.getvalue()
        print(csv_data)  # Output CSV content as a string
        csv_buffer = io.StringIO(csv_data)  # Convert string to a file-like object
        extract_post=pd.read_csv(csv_buffer)




        
        remove_duplicate = pd.DataFrame(extract_post)
        new_file = remove_duplicate.drop_duplicates(subset=['Id'])
        print(new_file.head())
        # new_file.to_csv("output_2.csv",index=False, header=True)
        papers = pd.DataFrame(new_file)  #  dataset of all unique tags  
# Remove punctuation
        papers = papers.replace('\n',' ', regex=True)   #replacew newline with space    
        papers['Title_text_processed'] = papers['Title'].map(lambda x: re.sub('[,\.!?]', ' ', x)) #remove punct!
        papers['Title_text_processed'] = papers['Title_text_processed'].map(lambda x: re.sub('<[^>]+>', ' ', x)) #remove html tag
        papers['Title_text_processed'] = papers['Title_text_processed'].map(lambda x: re.sub('[^a-zA-Z ]+', ' ', x)) #remove non alpahbetics  tag
        # Convert the titles to lowercase
        papers['Title_text_processed'] = papers['Title_text_processed'].map(lambda x: x.lower())
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

# Print out the first rows of papers
        print(papers.head())
        #Saving File  with distant posts 
        papers.to_csv('processed.csv',index=False,header=True)
        print(len(papers))
        proceed_post_file=papers.fillna(0)
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

# #         lda_model = gensim.models.LdaMulticore(corpus=corpus,
# #                                            id2word=id2word,
# #                                            num_topics=num_topics, 
# #                                            random_state=100,
# #                                            chunksize=100,
# #                                            passes=10,
# #                                            alpha=0.01,
# #                                            eta=0.91)
        

# Print the Keyword in the 10 topics
        # pprint(lda_model.print_topics())
        # doc_lda = lda_model[corpus]
        # #save Traing LADA model to CSV file 
        # PP = pd.DataFrame(lda_model.print_topics(-1,num_words=20))
        # PP.to_csv('Gensim_LDA1.csv',index=False,header=True)
        #Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                            num_topics=num_topics, 
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
        LDA_Topics_File=PP.fillna(0)
        
         
        # PP.to_csv('Gensim_LDA1.csv',index=False,header=True)
        # Compute Perplexity (inability to deal with or understand something)
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        # Calculate popularity metrics
        # Assign topics to posts using the LDA model
        topic_assignments = []
        for row in lda_model[corpus]:
                    dominant_topic = sorted(row[0], key=lambda x: x[1], reverse=True)[0][0]  # Get the most relevant topic
                    topic_assignments.append(dominant_topic)


        # Add topic assignments to the dataset
        papers['GeneratedTopic'] = topic_assignments
        topic_metrics = papers.groupby('GeneratedTopic').agg(
        ViewCount=('ViewCount', 'sum'),
        FavoriteCount=('FavoriteCount', 'sum'),
        Score=('Score', 'sum')).reset_index()

        # Normalize the metrics
        total_view = topic_metrics['ViewCount'].sum()
        total_favorite = topic_metrics['FavoriteCount'].sum()
        total_score = topic_metrics['Score'].sum()

        # Normalize the columns
        topic_metrics['ViewN'] = (topic_metrics['ViewCount'] * num_topics) / total_view
        topic_metrics['FavoriteN'] = (topic_metrics['FavoriteCount'] * num_topics) / total_favorite
        topic_metrics['ScoreN'] = (topic_metrics['Score'] * num_topics) / total_score

        # Compute the Fused Popularity Metric
        topic_metrics['FusedP'] = (topic_metrics['ViewN'] + topic_metrics['FavoriteN'] + topic_metrics['ScoreN']) / 3
        topic_metrics.to_csv("Papularity-by-topic.csv")
        papularity_of_topics=topic_metrics.fillna(0)
        
        df=papers
        # Step 1: Count total questions per topic
        total_questions = df.groupby("GeneratedTopic")["Id"].count().reset_index()
        total_questions.rename(columns={"Id": "TotalQuestions"}, inplace=True)
        df['IsAnswered'] = df['AcceptedAnswerId'].notnull().astype(int)
        papers['TimeToAnswer'] = (pd.to_datetime(papers['LastActivityDate']) - pd.to_datetime(papers['CreationDate'])).dt.total_seconds() / 3600

        # Step 2: Count unanswered questions per topic
        unanswered_questions = df[df["IsAnswered"] == 0].groupby("GeneratedTopic")["Id"].count().reset_index()
        unanswered_questions.rename(columns={"Id": "UnansweredQuestions"}, inplace=True)

        # Merge total and unanswered question counts
        difficulty_df = total_questions.merge(unanswered_questions, on="GeneratedTopic", how="left")
        difficulty_df["UnansweredQuestions"] = difficulty_df["UnansweredQuestions"].fillna(0)  # Fill NaN with 0

        # Step 3: Compute Percentage of Unanswered Questions
        difficulty_df["PctQWoAcceptedAnswer"] = (difficulty_df["UnansweredQuestions"] / difficulty_df["TotalQuestions"]) * 100

        # Step 4: Compute Median Hours to Get an Accepted Answer per Topic
        median_time_to_answer = df.groupby("GeneratedTopic")["TimeToAnswer"].median().reset_index()
        median_time_to_answer.rename(columns={"TimeToAnswer": "MedHrsToGetAccAns"}, inplace=True)

        # Merge with difficulty DataFrame
        difficulty_df = difficulty_df.merge(median_time_to_answer, on="GeneratedTopic", how="left")

        # Step 5: Normalize Metrics
        difficulty_df["PctQWoAcceptedAnswerNorm"] = (difficulty_df["PctQWoAcceptedAnswer"] * 40) / difficulty_df["PctQWoAcceptedAnswer"].sum()
        difficulty_df["MedHrsToGetAccAnsNorm"] = (difficulty_df["MedHrsToGetAccAns"] * 40) / difficulty_df["MedHrsToGetAccAns"].sum()

        # Step 6: Compute Final Fused Difficulty Score
        difficulty_df["FusedDifficulty"] = (difficulty_df["PctQWoAcceptedAnswerNorm"] + difficulty_df["MedHrsToGetAccAnsNorm"]) / 2

        # Display Results
        print(difficulty_df[["GeneratedTopic", "FusedDifficulty"]])

        # Save results to a CSV file (optional)
        difficulty_df.to_csv("topic_difficulty.csv", index=False)
        difficulty_metrics=difficulty_df.fillna(0)
       

        # Sample function to calculate entropy of tag distributiontopic_unanswered = df.groupby('Question_Category')['IsAnswered'].apply(lambda x: (x == 0).mean())

            # Compute median hours to get an accepted answer per topic category
            # Compute percentage of unanswered questions per topic category
        topic_unanswered = df.groupby('GeneratedTopic')['IsAnswered'].apply(lambda x: (x == 0).mean())

            # Compute median hours to get an accepted answer per topic category
        topic_median_time = df.groupby('GeneratedTopic')['TimeToAnswer'].median()

            # Normalize the values
        normalized_unanswered = (topic_unanswered * num_topics) / topic_unanswered.sum()
        normalized_median_time = (topic_median_time * num_topics) / topic_median_time.sum()

            # Compute the complexity score using the difficulty formula
        complexity_score = (normalized_unanswered + normalized_median_time) / 2

            # Combine results into a DataFrame
        complexity_df = pd.DataFrame({
                'PctQWoAcceptedAnswerNi': normalized_unanswered,
                'MedHrsToGetAccAnsNi': normalized_median_time,
                'Complexity': complexity_score
            }).sort_values(by='Complexity', ascending=False)

            # Display the computed complexity scores
        print(complexity_df)
        complexity_of_topics=complexity_df.fillna(0)
        
        # Trend analysis
        df['YearMonth'] = pd.to_datetime(df['CreationDate']).dt.to_period('M')

        trends_by_topic = df.groupby(['GeneratedTopic', 'YearMonth']).size().reset_index(name='Count')
        trends_by_topic['YearMonth'] = trends_by_topic['YearMonth'].astype(str)

        # Save results
        trends_by_topic.to_csv("trends_by_topic.csv", index=False)
        trends_of_topics=trends_by_topic.fillna(0)
         
        # Define function to categorize questions
        def categorize_question(title):
            title_lower = title.lower()
            if title_lower.startswith("what"):
                return "What"
            elif title_lower.startswith("why"):
                return "Why"
            elif title_lower.startswith("how"):
                return "How"
            else:
                return "Others"

        # Apply categorization to Title_text_processed column
        df['Question_Category'] =  df['Title_text_processed'].apply(categorize_question)

        # Calculate the count of each question type in each topic
        counts =  df.groupby(['GeneratedTopic', 'Question_Category']).size().unstack(fill_value=0)

        # Calculate the percentage of each question type in each topic
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100

        # Combine counts and percentages into a single dataframe
        result = counts.copy()
        for col in percentages.columns:
            result[f"{col}_Percentage"] = percentages[col]

        # Save or display the result
        print(result)
        result.to_csv("question_analysis_with_percentages.csv")  # Save the result to a file if needed
        question_analysis_with_percentages=result.fillna(0)
        
        # df["CreationDate"] = pd.to_datetime(df["CreationDate"], errors="coerce")

        # # Extract Year-Month
        # df["YearMonth"] = df["CreationDate"].dt.to_period("M")

        # # Compute Impact Score
        # df["Impact_Score"] = (df["ViewCount"] * 0.5) + (df["Score"] * 2) + (df["AnswerCount"] * 3) + (df["CommentCount"] * 1)

        # # 1. **Compute Absolute Impact**
        # def compute_absolute_impact(df):
        #     absolute_impact = df.groupby(["YearMonth", "GeneratedTopic"])["Impact_Score"].sum().reset_index()
        #     absolute_impact.rename(columns={"Impact_Score": "Absolute_Impact"}, inplace=True)
        #     return absolute_impact

        # absolute_impact_df = compute_absolute_impact(df)

        # # 2. **Compute Relative Impact**
        # def compute_relative_impact(df):
        #     post_counts = df.groupby("YearMonth")["Id"].nunique().reset_index()
        #     post_counts.rename(columns={"Id": "Total_Posts"}, inplace=True)
            
        #     relative_impact_df = compute_absolute_impact(df)
        #     relative_impact_df = relative_impact_df.merge(post_counts, on="YearMonth")
            
        #     relative_impact_df["Relative_Impact"] = relative_impact_df["Absolute_Impact"] / relative_impact_df["Total_Posts"]
            
        #     return relative_impact_df[["YearMonth", "GeneratedTopic", "Relative_Impact"]]

        # relative_impact_df = compute_relative_impact(df)

        # # 3. **Trend Shift Detection**
        # def detect_trend_shifts(df):
        #     df["Impact_Change"] = df.groupby("GeneratedTopic")["Absolute_Impact"].diff()
        #     df["Trend"] = df["Impact_Change"].apply(lambda x: "Increase" if x > 0 else "Decrease" if x < 0 else "Stable")
        #     return df[["YearMonth", "GeneratedTopic", "Absolute_Impact", "Impact_Change", "Trend"]]

        # trend_shifts_df = detect_trend_shifts(absolute_impact_df)

        # # 4. **Volatility Index**
        # def compute_volatility(df):
        #     volatility_df = df.groupby("GeneratedTopic")["Absolute_Impact"].std().reset_index()
        #     volatility_df.rename(columns={"Absolute_Impact": "Volatility"}, inplace=True)
        #     return volatility_df

        # volatility_df = compute_volatility(absolute_impact_df)

        # # 5. **Breakthrough Identification (Sudden Spikes)**
        # def identify_breakthroughs(df, threshold=2.0):
        #     df["Prev_Impact"] = df.groupby("GeneratedTopic")["Absolute_Impact"].shift(1)
        #     df["Breakthrough"] = (df["Absolute_Impact"] / df["Prev_Impact"]) > threshold
        #     return df[df["Breakthrough"]][["YearMonth", "GeneratedTopic", "Absolute_Impact"]]

        # breakthrough_df = identify_breakthroughs(absolute_impact_df)

        # # 6. **Relative Change Analysis (Month-over-Month)**
        # def compute_relative_change(df):
        #     df["Relative_Change"] = df.groupby("GeneratedTopic")["Absolute_Impact"].pct_change() * 100
        #     return df[["YearMonth", "GeneratedTopic", "Absolute_Impact", "Relative_Change"]]

        # relative_change_df = compute_relative_change(absolute_impact_df)

#         # Print results
#         print("Trend Shifts:\n", trend_shifts_df)
#         print("\nVolatility Index:\n", volatility_df)
#         print("\nBreakthroughs:\n", breakthrough_df)
#         print("\nRelative Change Analysis:\n", relative_change_df)
#         df.to_csv("finalize.csv")
         









#         df=pd.read_csv("finalize.csv")
#         df = df.fillna(0)  # Replace NaN with 0
        # Prepare the response
        response = {
            #"Topics": [{"index": index, "content": content} for index, content in lda_model],
            "Coherence Score":coherence_lda,
            "unique tags":unique_list_of_tags_file.to_dict(orient="records"),
            "significance and relevence of tags":new_list_of_tags_with_s_and_r_file.to_dict(orient="records"),
            "unique posts":uniqueposts_tags_data.to_dict(orient="records"),
            "total recomendation":df6.to_dict(orient="records"),
            "extracted post":proceed_post_file.to_dict(orient="records"),
            "Extracted Topics":LDA_Topics_File.to_dict(orient="records"),
            "papularity of topics":papularity_of_topics.to_dict(orient="records"),
            "difficulty of topics":difficulty_metrics.to_dict(orient="records"),
            "complexity_of_topics":complexity_of_topics.to_dict(orient="records"),
            "trends_of_topics":trends_of_topics.to_dict(orient="records"),
            "question_analysis_with_percentages":question_analysis_with_percentages.to_dict(orient="records"),
            
            
             
            "Perplexity":lda_model.log_perplexity(corpus)

        }

        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from typing import List
# import re
# from gensim.corpora import Dictionary
# from gensim import corpora
# from gensim.models.ldamodel import LdaModel
# from datetime import datetime

# app = FastAPI()

# # Allow CORS for frontend integration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Helper function to preprocess text
# def preprocess_text(text: str) -> str:
#     text = re.sub('[^a-zA-Z]', ' ', text)
#     text = text.lower()
#     text = re.sub(r"\s+", " ", text)
#     return text

# # Endpoint to analyze topics, popularity, and difficulty
# @app.post("/analyze/")
# async def analyze_topics(
#     file: UploadFile,
#     text_column: str = Form(...),
#     score_column: str = Form(...),
#     view_column: str = Form(...),
#     favorite_column: str = Form(...),
#     answer_column: str = Form(...),
#     accepted_column: str = Form(...),
#     time_column: str = Form(...),
#     num_topics: int = Form(10)
# ):
#     try:
#         # Load dataset
#         data = pd.read_csv(file.file)

#         # Preprocess text data
#         if text_column not in data.columns:
#             return JSONResponse(content={"error": f"'{text_column}' column not found."}, status_code=400)
#         data[text_column] = data[text_column].apply(preprocess_text)

#         # LDA Topic Modeling
#         vectorizer = CountVectorizer(stop_words="english")
#         data_vectorized = vectorizer.fit_transform(data[text_column])
#         lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#         lda_model.fit(data_vectorized)
        
#         # Extract topics and top words
#         words = vectorizer.get_feature_names_out()
#         topics = {f"Topic {i+1}": [words[j] for j in topic.argsort()[-10:]] for i, topic in enumerate(lda_model.components_)}

#         # Assign topics to documents
#         topic_assignments = lda_model.transform(data_vectorized)
#         data['Topic'] = topic_assignments.argmax(axis=1) + 1

#         # Calculate Popularity Metrics
#         popularity_metrics = data.groupby('Topic').agg({
#             score_column: 'mean',
#             view_column: 'mean',
#             favorite_column: 'mean',
#             answer_column: 'mean'
#         }).rename(columns={
#             score_column: 'Avg_Score',
#             view_column: 'Avg_Views',
#             favorite_column: 'Avg_Favorites',
#             answer_column: 'Avg_Answers'
#         }).reset_index()

#         # Calculate Difficulty Metrics
#         data[accepted_column] = data[accepted_column].fillna(False).astype(bool)
#         data[time_column] = pd.to_datetime(data[time_column])

#         difficulty_metrics = data.groupby('Topic').apply(lambda x: pd.Series({
#             '%_Accepted_Answers': x[accepted_column].mean() * 100,
#             'Avg_Time_To_Accepted': x.loc[x[accepted_column], time_column].apply(lambda t: (t - x[time_column].min()).dt.total_seconds()).mean() / 3600 if not x[accepted_column].empty else None,
#             '%_Unanswered': (x[answer_column] == 0).mean() * 100
#         })).reset_index()

#         # Merge Metrics
#         metrics = pd.merge(popularity_metrics, difficulty_metrics, on='Topic', how='left')

#         # Return Results
#         response = {
#             "Topics": topics,
#             "Metrics": metrics.to_dict(orient='records')
#         }
#         return JSONResponse(content=response, status_code=200)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

