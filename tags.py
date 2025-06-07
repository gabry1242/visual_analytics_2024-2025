import pandas as pd

# Load the dataset
df = pd.read_csv("merged_with_tags.csv")  # or use the dataframe you already have

# Combine all tags into a single list
all_tags = df['tag'].dropna().str.split(';').explode().str.strip().str.lower()

# Get the set of unique tags
unique_tags = set(all_tags)

# Count the occurrences of each tag
tag_counts = all_tags.value_counts()

# Filter tags that appear more than 5 times
popular_tags = tag_counts[tag_counts > 100]

# Display the result
print(popular_tags)

# Count of unique tags
print(len(unique_tags))