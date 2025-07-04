#!/usr/bin/env python
# coding: utf-8

# <h3><center>Entity Duplication Detection</center></h3>
# <h3><center>26th April - 2nd March</center></h3>

# ##### 1. Data Exploration - Initial Findings

# In[24]:


#Import Required Libraries and Packages.

import pandas as pd
import re   # Regular Expression
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from IPython.display import display, HTML
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer


# In[25]:


# Load the dataset

data = "QUB_Analytathon2_Deloitte_data.csv"
df = pd.read_csv(data)


# In[26]:


# Display basic information about the dataset

df_info = df.info()
df_head = df.head() 
df_missing_values = df.isnull().sum()

df_info, df_head, df_missing_values


# ##### 2. Data Exploration - Initial Findings

# In[27]:


# Standardize date_of_birth format (convert to datetime, handle missing values)

df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')


# In[28]:


# Normalize email addresses (lowercase, strip spaces)

df['email_address'] = df['email_address'].str.lower().str.strip()


# In[29]:


# Standardize phone numbers (remove spaces, dashes, and normalize country codes)

def clean_phone_number(phone):
    if pd.isna(phone):
        return None
    phone = re.sub(r'\D', '', phone)  # Remove non-numeric characters
    if phone.startswith("44") and len(phone) > 10:  # Convert UK numbers (remove country code)
        phone = "0" + phone[2:]
    return phone

df['phone_number'] = df['phone_number'].apply(clean_phone_number)

# Standardize name fields (lowercase, remove extra spaces)

df['first_name'] = df['first_name'].str.lower().str.strip()
df['middle_name'] = df['middle_name'].str.lower().str.strip()
df['last_name'] = df['last_name'].str.lower().str.strip()


# In[30]:


# Remove leading/trailing spaces from address fields

address_cols = ['house_no', 'primary_street', 'town', 'postcode', 'county']
for col in address_cols:
    df[col] = df[col].astype(str).str.strip().replace("nan", None)


# In[31]:


# Check how many records still have missing key values

missing_values_after_cleaning = df.isnull().sum()
missing_values_after_cleaning.head(12)


# ##### 3.1: Exact Duplicate Detection - Results

# In[32]:


# Identify Exact Duplicates using strong identifiers (email_address, phone_number)

# Finding exact duplicates based on email_address (ignoring missing values)
exact_duplicates_email = df[df.duplicated(subset=['email_address'], keep=False) & df['email_address'].notna()]

# Finding exact duplicates based on phone_number (ignoring missing values)
exact_duplicates_phone = df[df.duplicated(subset=['phone_number'], keep=False) & df['phone_number'].notna()]

# Count of exact duplicates found
num_exact_email_dupes = exact_duplicates_email.shape[0]
num_exact_phone_dupes = exact_duplicates_phone.shape[0]

print(" No of exact email duplicates found: ", num_exact_email_dupes,"\n","No of exact phone number duplicates found: ", num_exact_phone_dupes)


# ##### 3.2: Fuzzy Matching - Results

# In[33]:


# Fuzzy Matching - Name & Address Similarity

# Define a function to compare two strings using Levenshtein Distance (fuzz ratio)
def fuzzy_match(str1, str2, threshold=85):
    if pd.isna(str1) or pd.isna(str2):
        return False
    return fuzz.ratio(str1, str2) >= threshold


# In[34]:


# Create a new dataframe for potential fuzzy duplicates
potential_fuzzy_duplicates = []


# In[35]:


# Compare each record with others (this is computationally expensive, so we limit to subsets)
df_subset = df[['first_name', 'last_name', 'date_of_birth', 'email_address', 'phone_number', 'primary_street', 'town', 'postcode']].dropna()


# In[36]:


# Iterate over pairs of records (this can be optimized with clustering techniques later)
for i, row1 in df_subset.iterrows():
    for j, row2 in df_subset.iterrows():
        if i >= j:  # Avoid redundant comparisons
            continue
        
        # Apply fuzzy matching on names and addresses
        name_match = fuzzy_match(row1['first_name'], row2['first_name']) and fuzzy_match(row1['last_name'], row2['last_name'])
        address_match = fuzzy_match(row1['primary_street'], row2['primary_street']) and fuzzy_match(row1['town'], row2['town'])
        
        # Consider as duplicate if name AND address match with high similarity
        if name_match and address_match:
            potential_fuzzy_duplicates.append((i, j))

# Count potential fuzzy duplicates found
num_fuzzy_dupes = len(potential_fuzzy_duplicates)
print("No of potential fuzzy duplicates found :",num_fuzzy_dupes)


# ##### 4. Preventing Future Duplicates

# In[37]:


# Combine exact and fuzzy duplicates into a final report

# Convert exact duplicates into sets of indexes
exact_dupes_indexes = set(exact_duplicates_email.index).union(set(exact_duplicates_phone.index))

# Convert fuzzy duplicates into sets of indexes
fuzzy_dupes_indexes = set(i for pair in potential_fuzzy_duplicates for i in pair)


# In[38]:


# Merge both sets
final_duplicate_indexes = exact_dupes_indexes.union(fuzzy_dupes_indexes)


# In[39]:


# Extract the final duplicate records from the original dataset
final_duplicate_records = df.loc[list(final_duplicate_indexes)]


# In[40]:


# Save the duplicate records to a CSV file for review
output_file = "detected_duplicates.csv"
final_duplicate_records.to_csv(output_file, index=False)


# In[41]:


# Display number of total duplicates found and provide download link
num_total_dupes = len(final_duplicate_records)
print("Number of total duplicates found: ", num_total_dupes, output_file)


# In[42]:


# Creating a summary dictionary
summary_data = {
    "Category": [
        "Total Records Processed",
        "Missing Middle Name",
        "Missing Date of Birth",
        "Missing Email/Phone",
        "Exact Duplicates (Phone-based)",
        "Exact Duplicates (Email-based)",
        "Fuzzy Duplicates (Name & Address)",
        "Total Duplicates Identified"
    ],
    "Count": [
        5000,  # Assuming total records
        "49% missing",
        "63% missing",
        "~36% missing",
        2837,
        1565,
        104,
        3512
    ]
}

# Convert dictionary to DataFrame
df_summary = pd.DataFrame(summary_data)
df_summary


# ##### Extra code: Creating a separate HTML file to display the output

# In[43]:


# Creating a styled HTML table with CSS
styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
        }}
        table {{
            width: 60%;
            border-collapse: collapse;
            margin: auto;
            font-size: 18px;
            text-align: center;
        }}
        th {{
            background-color: #007BFF;
            color: white;
            padding: 12px;
            text-align: center;
        }}
        td {{
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #ddd;
        }}
    </style>
</head>
<body>
    <h2 style="text-align:center;">Duplicate Detection Summary</h2>
    {df_summary.to_html(index=False, escape=False)}
</body>
</html>
"""

# Save the styled HTML file
styled_html_path = "duplicate_summary_styled.html"
with open(styled_html_path, "w", encoding="utf-8") as f:
    f.write(styled_html)

# Generate a clickable link to open the styled HTML summary
html_link = f'<a href="{styled_html_path}" target="_blank" style="font-size: 18px; color: blue; text-decoration: underline;">Click here to view the summary</a>'

# Display the clickable link in the notebook output
display(HTML(html_link))


# ##### 
# ##### Lets create the following visualizations:
# 
# ##### 1️ Missing Data Overview → Percentage of missing values in key fields
# ##### 2️ Duplicate Distribution → Proportion of exact vs. fuzzy duplicates
# ##### 3️ Top Fields Causing Duplicates → Phone, Email, Name, Address contributions
# ##### 4️ Data Entry Issues → Frequency of email/phone format errors

# In[44]:


# Set style
sns.set(style="whitegrid")

# Sample data
missing_data = {"Middle Name": 49, "Date of Birth": 63, "Email/Phone": 36}
duplicate_distribution = {"Exact Duplicates (Phone/Email)": 4402, "Fuzzy Duplicates (Name & Address)": 104}
duplicate_causes = {"Phone Number": 2837, "Email Address": 1565, "Name & Address": 104}

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1️ Missing Data Percentage (Bar Chart)
sns.barplot(x=list(missing_data.keys()), y=list(missing_data.values()), ax=axes[0, 0], hue=list(missing_data.keys()), legend=False, palette="Reds_r")
axes[0, 0].set_title("Missing Data Percentage", fontsize=14)
axes[0, 0].set_ylabel("Percentage (%)", fontsize=12)
axes[0, 0].set_xticks(range(len(missing_data)))  # Set fixed tick locations
axes[0, 0].set_xticklabels(list(missing_data.keys()), fontsize=11)  # Now safely set labels

# Add bar labels
for bar in axes[0, 0].containers:
    axes[0, 0].bar_label(bar, fmt='%d%%', fontsize=10)

# Explanation Below Graph
axes[0, 0].text(-0.3, -10, "Date of Birth has 63% missing values, making identity matching difficult.", fontsize=11, ha="left", color="black")
axes[0, 0].text(-0.3, -15, "Email/Phone is missing in 36% of records, reducing duplicate detection accuracy.", fontsize=11, ha="left", color="black")

# 2️ Duplicate Distribution (Pie Chart)
axes[0, 1].pie(duplicate_distribution.values(), labels=duplicate_distribution.keys(), autopct='%1.1f%%', colors=["#66b3ff", "#ff9999"], textprops={'fontsize': 11})
axes[0, 1].set_title("Duplicate Record Distribution", fontsize=14)

# Explanation Below Graph
axes[0, 1].text(-1.5, -1.4, "Most duplicates (4402 records) are exact matches based on phone/email.", fontsize=11, ha="left", color="black")
axes[0, 1].text(-1.5, -1.55, "Fuzzy duplicates (104 records) suggest minor differences in name or address.", fontsize=11, ha="left", color="black")

# 3️ Top Fields Causing Duplicates (Bar Chart)
sns.barplot(x=list(duplicate_causes.keys()), y=list(duplicate_causes.values()), ax=axes[1, 0], hue=list(duplicate_causes.keys()), legend=False, palette="coolwarm_r")
axes[1, 0].set_title("Top Fields Causing Duplicates", fontsize=14)
axes[1, 0].set_ylabel("Number of Duplicates", fontsize=12)
axes[1, 0].set_xticks(range(len(duplicate_causes)))  # Set fixed tick locations
axes[1, 0].set_xticklabels(list(duplicate_causes.keys()), fontsize=11)  # Now safely set labels

# Add bar labels
for bar in axes[1, 0].containers:
    axes[1, 0].bar_label(bar, fmt='%d', fontsize=10)

# Explanation Below Graph
axes[1, 0].text(-0.3, -500, "Phone Number is the most common duplication source, causing 2837 duplicate records.", fontsize=11, ha="left", color="black")
axes[1, 0].text(-0.3, -700, "Email Address is the second biggest factor, contributing to 1565 duplicates.", fontsize=11, ha="left", color="black")

# Hide unused subplot
axes[1, 1].axis("off")

# Adjust layout for better spacing
plt.tight_layout()

# Show the updated visualization
plt.show()


# In[46]:


import pandas as pd
from rapidfuzz import fuzz
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

# Load dataset
df = pd.read_csv("detected_duplicates.csv")

# Function for fuzzy matching
def fuzzy_match(str1, str2, threshold=85):
    if pd.isna(str1) or pd.isna(str2):
        return False
    return fuzz.ratio(str1, str2) >= threshold

# Identifying possible duplicates
potential_duplicates = []
for idx1, idx2 in combinations(df.index, 2):
    row1, row2 = df.loc[idx1], df.loc[idx2]
    
    name_match = fuzzy_match(row1['first_name'], row2['first_name']) and \
                 fuzzy_match(row1['last_name'], row2['last_name'])
    email_match = row1['email_address'] == row2['email_address'] if not pd.isna(row1['email_address']) else False
    phone_match = row1['phone_number'] == row2['phone_number'] if not pd.isna(row1['phone_number']) else False
    
    if name_match or email_match or phone_match:
        potential_duplicates.append((idx1, idx2))

# Creating a DataFrame with duplicate pairs
duplicate_pairs = pd.DataFrame(potential_duplicates, columns=['Record 1', 'Record 2'])

# ML-Based Clustering for Duplicate Detection
vectorizer = TfidfVectorizer(stop_words='english')
text_features = df[['first_name', 'last_name', 'email_address']].fillna('').agg(' '.join, axis=1)
X = vectorizer.fit_transform(text_features)

# Applying DBSCAN clustering
clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(X.toarray())
df['Cluster'] = clustering.labels_

# Display summary
duplicate_summary = {
    "Total Records": len(df),
    "Potential Duplicates Found (Rule-Based)": len(duplicate_pairs),
    "Clusters Detected (ML-Based)": len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
}

duplicate_summary


# In[55]:





# In[ ]:




