#!/usr/bin/env python
# coding: utf-8

# # Duplication Detection

# In[279]:


# Import libraries and packages
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz, process
import recordlinkage
import matplotlib.pyplot as plt


# ## Exact Duplication Detection

# In[281]:


# Load cleaned dataset
df = pd.read_csv("cleaned_deloitte_data.csv")

# Identify exact duplicates based on phone number and email address as these are unique identifiers

# Finds duplicates based on email address, ensures missing emails are not included 
exact_duplicates_email = df[df.duplicated(subset=["email_address"], keep=False) & df["email_address"].notna()]

# Finds duplicates based on phone number, ensures missing phone numbers are not included 
exact_duplicates_phone = df[df.duplicated(subset=["phone_number"], keep=False) & df["phone_number"].notna()]

# Combine exact duplicates
# Drops duplicate rows in the combined dataset since a record may appear in both email and phone duplicates
exact_duplicates = pd.concat([exact_duplicates_phone, exact_duplicates_email]).drop_duplicates()

# Print number of exact duplicates found
print(f"Exact phone-based duplicates: {exact_duplicates_phone.shape[0]}")
print(f"Exact email-based duplicates: {exact_duplicates_email.shape[0]}")
print(f"Total exact duplicates: {exact_duplicates.shape[0]}")

# Remove exact duplicates before Fuzzy Matching
df_remaining = df.drop(exact_duplicates.index)

# Print number of records remaining after exact duplicates removed
print(f"Records remaining after removing exact duplicates: {df_remaining.shape[0]}")


# In[283]:


# Load original dataset
original_df = pd.read_csv('QUB_Analytathon2_Deloitte_data.csv')

# Count of exact duplicates based on email and phone
duplicates_count = {
    'Total Records': original_df.shape[0], # Total number of original records before removing duplicates 
    'Exact Phone Duplicates': exact_duplicates_phone.shape[0],  # Count of duplicates based on phone number
    'Exact Email Duplicates': exact_duplicates_email.shape[0],  # Count of duplicates based on email address
    'Total Exact Duplicates': exact_duplicates.shape[0],  # Total number of exact duplicates (email + phone)
}

# Create a bar plot to visualize the count of exact duplicates and original records
plt.figure(figsize=(8, 5))  # Set the size of the plot
bars = plt.bar(duplicates_count.keys(), duplicates_count.values(), color=['blue', 'blue', 'blue', 'blue'])

# Add a title to the plot
plt.title('Count of Exact Duplicates and Original Records', fontsize=16) # Title
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45) # Rotate x axis for readibility

# Add the values on top of the bars for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=12)

plt.show()



# ## Fuzzy Matching

# In[285]:


# Initialize the indexer for record linkage
indexer = recordlinkage.Index()

# Use sorted neighbourhood blocking strategy on "first_name" to group similar records together (alphabetical)
# Significantly reduces the number of comparisons as every record isnt compared with every other record - much faster
indexer.sortedneighbourhood("first_name")  # Blocks based on first_name proximity

# Create pairs of candidate records to compare
pairs = indexer.index(df_remaining)

# Initialize a comparison object to perform fuzzy matching on the selected pairs
compare = recordlinkage.Compare()

# Fuzzy matching on "first_name" column using the Jaro-Winkler similarity method
# Only pairs with at least 85% similarity in their first names will be considered a match
compare.string("first_name", "first_name", method="jarowinkler", threshold=0.8, label="first_name_sim")

# Fuzzy matching on "last_name" column using the Jaro-Winkler similarity method
# Only pairs with at least 85% similarity in their last names are considered a match
compare.string("last_name", "last_name", method="jarowinkler", threshold=0.8, label="last_name_sim")

# Fuzzy matching on "primary_street" column using the Levenshtein similarity method
# Only pairs with at least 80% similarity in their street addresses are considered a match
compare.string("primary_street", "primary_street", method="levenshtein", threshold=0.8, label="street_sim")

# Fuzzy matching on "town" column using the Levenshtein similarity method
# Only pairs with at least 80% similarity in their street addresses are considered a match
compare.string("town", "town", method="levenshtein", threshold=0.8, label="town_sim")

# Compute the similarity scores for all the candidate pairs that were identified 
fuzzy_matches = compare.compute(pairs, df_remaining)

# Calculate a weighted similarity score using the individual similarity scores from the fuzzy matching
# Here, first_name and last_name are given a weight of 0.4 each, while street is given a weight of 0.2
fuzzy_matches["similarity_score"] = (
    fuzzy_matches["first_name_sim"] * 0.4 + 
    fuzzy_matches["last_name_sim"] * 0.4 + 
    fuzzy_matches["street_sim"] * 0.1 +
    fuzzy_matches["town_sim"] * 0.1            #added this in to see what it did, not necessarily gunna keep it 
)

# Set the threshold for the overall similarity score
final_threshold = 0.80

# Identify potential fuzzy duplicates (0.8 <= similarity score < 1.0)
potential_fuzzy_duplicates = fuzzy_matches[
    (fuzzy_matches["similarity_score"] >= final_threshold) & (fuzzy_matches["similarity_score"] < 1.0)
]

# Identify likely fuzzy duplicates (similarity score == 1.0)
fuzzy_duplicates = fuzzy_matches[fuzzy_matches["similarity_score"] == 1.0]

# Display the results
print(f"Number of fuzzy duplicates found: {fuzzy_duplicates.shape[0]}") # highly likely duplicates
print(f"Number of potential fuzzy duplicates found: {potential_fuzzy_duplicates.shape[0]}") # some are duplicates, some not

# in report i will talk about how its not perfectly accurate, but those w 1.0 similarity score are defo duplicates 



# In[287]:


# Count occurrences of each similarity score
score_counts = fuzzy_matches["similarity_score"].value_counts().sort_index()

# Set up figure size
plt.figure(figsize=(10, 6))

# Create bar chart
plt.bar(score_counts.index, score_counts.values, color="blue", width=0.1)

# Add title and labels
plt.title("Distribution of Similarity Scores for Fuzzy Duplicates", fontsize=16)
plt.xlabel("Similarity Score", fontsize=12)
plt.ylabel("Number of Record Pairings", fontsize=12)

# Add values on top of bars
for i, v in enumerate(score_counts.values):
    plt.text(score_counts.index[i], v + 2, str(v), ha="center", fontsize=14)

# Show plot
plt.show()


#think this looks better than the histogram from the presentation, havent decided whether to use histogram or not, histogram is at bottom
#of notebook but it doesnt really correlate to 100% to the actual records as far as whether they are duplicates or not, the threshold
# catches some but not all duplicates and flags some records that arent duplicates, thats why i decided to change it to possible duplicates
# vs highly likely and have the below graph, sim score of 1 are duplicates and for sim score of 0.8, some are and some arent



# In[289]:


# Plot Pie Chart of Duplicates Breakdown
labels = ['Exact Duplicates', 'Fuzzy Duplicates', 'Potential Fuzzy Duplicates']
sizes = [exact_duplicates.shape[0], fuzzy_duplicates.shape[0], potential_fuzzy_duplicates.shape[0]]
colors = ['lightblue','red', 'pink']
#explode = (0.1, 0)  # Slightly separate exact duplicates slice

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=False)
plt.title("Percentage of Duplicates Detected", size = 14)
plt.show()


# In[ ]:




