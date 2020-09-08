import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_aAa.csv')
titles = df.values[:, 2]
y = df.values[:, -1]
print(titles)
print(y)

titles_train, titles_test, y_train, y_test = train_test_split(
    titles, y, test_size=0.25, random_state=1410)

# Extractor
extractor = CountVectorizer().fit(titles_train)

print("Extracted")
