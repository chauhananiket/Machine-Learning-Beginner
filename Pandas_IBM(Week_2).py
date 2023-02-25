import pandas as pd

df = pd.read_csv(r'F:\Python Programs - ML\Dataset\Unzipped\CSV Files\olympics.csv',index_col=0, skiprows=1)
print(df.head())

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
      df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)
print(df.head())                           
                         
names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
print(df.head(3)) 
print(df.tail(3)) 
print(df)
print(df.shape)

print(df[1:3])

def answer_one():
    return df.index[df['Gold']==df['Gold'].max()] 
print(answer_one())

