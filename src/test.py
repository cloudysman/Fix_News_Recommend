import pandas as pd
news_data = pd.read_table(r'D:\HieuDT\news-recommendation\src\data\val\news_parsed.tsv')
#print(news_data[news_data['id'] == 'N55468'])?
print(news_data['id'] == "N55468")