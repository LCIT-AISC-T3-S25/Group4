# 📊 Project Task Allocation

## 🔹 Section A: For the given data write and run regex in your jupyter notebook to answer the following questions. Mention your answers clearly. (60%)

| **Question**                                                                        | **Assigned To** | **LLM Prompt Used**                                                                                                                 | **Result**                                           |
| ----------------------------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Q1. How many records have a date that is expressed without using alphabets?         | Aravind         | *"Using pandas and regex, count records with date formats like 'dd-mm-yyyy' or 'yyyy/mm/dd' that contain only digits and symbols."* | *0* |
| Q2. How many records have a word starting with the letter “w”?                      | Dhruv           | *"Write regex in pandas to count records containing a word that starts with lowercase or uppercase 'w'. Use `\bw\w*`."*             | *(e.g., 89 records)*                                 |
| Q3. How many records make a word that starts with an alphabet and is not a URL?     | Kabir           | *"Count records with a word starting with an alphabet but not starting with 'http' or 'www' using pandas and regex."*               | *(e.g., 213 records)*                                |
| Q4. How many tweets contain one of these emojis :), \:D, ;), \:P?                   | Preet           | *"In pandas, count tweets containing ':)', '\:D', ';)', or '\:P' using regex with escape characters."*                              | *(e.g., 47 tweets)*                                  |
| Q5. How many records contain a decimal number?                                      | Aesha           | *"Use regex in pandas to match decimal numbers like '3.14' or '0.99' using pattern `\d+\.\d+`."*                                    | *(e.g., 76 records)*                                 |
| Q6. What is the total number of IP addresses across all the records?                | Pranay          | *"Use regex to extract all IPv4 patterns `\b\d{1,3}(\.\d{1,3}){3}\b` and count total matches across dataframe."*                    | *(e.g., 105 IPs)*                                    |
| Q7. How many records have a new line?                                               | Nidhi           | *"In pandas, use `str.contains('\n')` to count records that include newline characters."*                                           | *(e.g., 18 records)*                                 |
| Q8. What is the total number of hashtags across all these tweets?                   | Smit            | *"Use regex `#\w+` to extract all hashtags in pandas column and sum total across all records."*                                     | *(e.g., 412 hashtags)*                               |
| Q9. What is the code to substitute all non-alphanumeric characters with a new line? | Nithish         | *"Use `re.sub(r'[^A-Za-z0-9]', '\\n', text)` to replace non-alphanumeric characters with newline."*                                 | `re.sub(r'[^A-Za-z0-9]', '\n', text)`                |
| Q10. What is the total number of URLs across all tweets?                            | Taran           | *"Use regex `http[s]?://\S+` or `www\.\S+` to extract and count total URLs in dataset."*                                            | *(e.g., 132 URLs)*                                   |

---

## 🔹 Section B: Perform stemming and lemmatization (40%)


| **Question** | **Assigned Team**   |
|--------------|---------------------|
| Q1  Use porter stemmer to run stemming. Count the number of uniquewords/tokens before and after stemming         | Kabir / Taran       |
| Q2  Perform lemmatization using NLTK lemmatizer. Count the number of unique words/tokens before and after lemmatization       | Dhruv / Aravind     |
| Q3  Compare the change in word frequencies from stemming and lemmatization. Which are the top 10 words after stemming/lemmatization         | Smit / Preet        |
| Q4  What is the change in word frequencies if normalization is done after stop word removal.        | Nidhi / Aesha       |
