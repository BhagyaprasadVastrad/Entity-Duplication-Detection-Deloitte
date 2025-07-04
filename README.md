# Analytathon 2 – Entity Duplication Detection

## Project Overview

In this project, I developed a data deduplication model to identify and remove duplicate records. My approach leveraged data preprocessing, fuzzy matching, and thresholding to detect duplicates. The goal was to enhance data quality and improve business efficiency.

## Author

Bhagyaprasad Vastrad

## Data Preprocessing

I performed the following preprocessing steps to prepare the data for duplicate detection:

- Standardised the `date_of_birth` field.
- Normalised `email_address` by converting to lowercase and removing spaces.
- Standardised `phone_number` by removing special characters and ensuring uniformity.


![image](https://github.com/user-attachments/assets/f29167fe-1d15-44a8-aba9-8518b466a6ea)

![image](https://github.com/user-attachments/assets/857a81fa-f4a0-4c5f-8922-bb246e380315)

## Duplicate Detection Approach

1. **Removed Exact Duplicates:** I identified and eliminated records with the same email or phone number.
2. **Generated Potential Matches:** I applied Sorted Neighborhood Indexing on first names to efficiently pair similar records.
3. **Calculated Similarity Scores:** I assigned weighted scores based on:
   - First Name Similarity (40%)
   - Last Name Similarity (40%)
   - Primary Street Similarity (20%)
4. **Filtered Duplicates:** I considered records with a similarity score above 0.75 as probable duplicates.

## Fuzzy Matching

I used fuzzy matching to identify similar but not identical records within the dataset. This technique helped me detect duplicates even when there were minor differences in names, addresses, or contact details, which is essential for handling inconsistencies caused by typos, varying formats, and incomplete data.

### Similarity Threshold

- The similarity score ranged from 0 (completely different) to 1 (identical).
- I set a threshold of 0.75, meaning only records with 75% or higher similarity were considered potential duplicates.
- I ensured that the threshold was neither too low (e.g., 0.6, which could incorrectly label unique records as duplicates) nor too high (e.g., 0.95, which could miss actual duplicates).

![image](https://github.com/user-attachments/assets/9c7bbca8-9fd4-44a3-903c-91e965b78b4d)

## Key Findings

- I detected 3,616 duplicates out of 5,186 records, meaning over 75% of the records were duplicates.
- The majority (3,512 records) were exact matches based on phone numbers and email addresses.
- I identified a further 104 duplicates using fuzzy matching.
- I faced challenges due to a significant amount of missing data: 63% missing in date of birth and 36% missing in email/phone fields, which complicated the analysis.

## Solutions Recommended

Based on my findings, I recommend:

- Implementing standardised data entry to ensure uniform formatting for names, phone numbers, and addresses.
- Introducing real-time duplicate detection to validate customer data during entry.
- Using unique identifiers such as customer IDs instead of relying solely on names.

## Conclusion

By combining data preprocessing, rule-based exact matching, and weighted fuzzy matching, I effectively detected duplicates and improved data quality, enabling better business decisions.
