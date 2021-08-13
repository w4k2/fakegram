import os
import pandas as pd

main_path = 'articles'
dates = os.listdir(main_path)

all_files = []
for root, dirs, files in os.walk(main_path):
    for file in files:
        all_files.append(os.path.join(root, file))

articles = []
titles = []
sources = []
for file in all_files:
    current_file = open(file, "r")
    article = current_file.read()
    current_file.close()

    articles.append(article)

    splited_path = file.split("\\")
    title_with_date = splited_path[-1]
    title = title_with_date.split("--")[-1]
    titles.append(title)

    source = splited_path[-2]
    sources.append(source)

dictionary = {}
dictionary['titles'] = titles
dictionary['articles'] = articles
dictionary['sources'] = sources

df = pd.DataFrame(data=dictionary)

df.to_csv('data/joined_data.csv', index=False)
