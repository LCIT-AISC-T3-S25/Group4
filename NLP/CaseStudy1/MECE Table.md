# 📊 MECE Table

## 🔹 Section A: For the given data write and run regex in your jupyter notebook to answer the following questions. Mention your answers clearly. (60%)

| **Question**                                                                        | **Assigned To** | **LLM Prompt Used**                                                                                                                 |
| ----------------------------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Q1. How many records have a date that is expressed without using alphabets?         | Aravind         | *"How can I count how many records in a pandas column have dates formatted like dd/mm/yyyy or yyyy-mm-dd without any alphabets?, Give me a regular expression to match dates written as numbers only, like dd/mm/yyyy or yyyy-mm-dd, in a pandas Series."* |
| Q2. How many records have a word starting with the letter “w”?                      | Dhruv           | *"How	can	I	count	the	number	of	rows	in	pandas	where	the	text	column	contains	a	word	starting	with	the	letter	'w'? Explain	what	the	regex	pattern	\b[wW]\w*	means	and	why	it	is	used	to	detect	words	starting	with	'w'	or	'W'."             |
| Q3. How many records make a word that starts with an alphabet and is not a URL?     | Kabir           | *"Count records with a word starting with an alphabet but not starting with 'http' or 'www' using pandas and regex."*               |
| Q4. How many tweets contain one of these emojis :), \:D, ;), \:P?                   | Preet           | *"In pandas, count tweets containing ':)', '\:D', ';)', or '\:P' using regex with escape characters."*                              |
| Q5. How many records contain a decimal number?                                      | Aesha           | *"Use regex in pandas to match decimal numbers like '3.14' or '0.99' using pattern `\d+\.\d+`."*                                    |
| Q6. What is the total number of IP addresses across all the records?                | Pranay          | *"Use regex to extract all IPv4 patterns `\b\d{1,3}(\.\d{1,3}){3}\b` and count total matches across dataframe."*                    |
| Q7. How many records have a new line?                                               | Nidhi           | *"How to find how many records have a new line using pandas, and explain all steps in detail."*                                           |
| Q8. What is the total number of hashtags across all these tweets?                   | Smit            | *"How can I count how many hashtags in my excel file. How can I count only unique hashtags"*                                     |
| Q9. What is the code to substitute all non-alphanumeric characters with a new line? | Nithish         | *"Use `re.sub(r'[^A-Za-z0-9]', '\\n', text)` to replace non-alphanumeric characters with newline."*                                 |
| Q10. What is the total number of URLs across all tweets?                            | Taran           | *"How to clean text data by converting to lowercase and removing extra whitespace using pandas? & what is the regex used for url?
"*                                            |

---

## 🔹 Section B: Perform stemming and lemmatization (40%)


| **Question**                                                                                                                         | **Assigned Team** | **LLM Prompt Used**                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Q1. Use Porter Stemmer to run stemming. Count the number of unique words/tokens before and after stemming.                           | Kabir / Taran     | *"Taranjot :- How to tokenize text data ? & How can I perform Stemming on the list of tokens? 
Kabir :- How to count the number of unique words in a pandas ? & What is the use of explode in stemming?"*                    |
| Q2. Perform lemmatization using NLTK lemmatizer. Count the number of unique words/tokens before and after lemmatization.             | Dhruv / Aravind   | *"Aravind - How do i initialize both stemmer and lemmatizer and set the data path manually for customer nltk_data, how can i manualy download the punkt tokeniker, how can i tokenize each row using nltk word tokenize funtion. Dhurv - how can i apply nltk word lematizer to each token in pandas column, how do i count the number of unique words before and after lemmatization"*                      |
| Q3. Compare the change in word frequencies from stemming and lemmatization. Which are the top 10 words after stemming/lemmatization? | Smit / Preet      | *"I want to count word frequencies from stemming and want to print it. last:And also I want for Lemmatization."* |
| Q4. What is the change in word frequencies if normalization is done after stop word removal?                                         | Nidhi / Aesha     | *"Nidhi - What is the change in word frequencies normalization is done after stop word removal and  how can i implatement this in pandas. Aesha -"* |
