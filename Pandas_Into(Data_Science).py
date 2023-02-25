import pandas as pd
#Creating dataframe
weather_data = {
    'day': ['1/1/2017','1/2/2017','1/3/2017','1/4/2017','1/5/2017','1/6/2017'],
    'temperature': [32,35,28,24,32,31],
    'windspeed': [6,7,2,7,4,2],
    'event': ['Rain', 'Sunny', 'Snow','Snow','Rain', 'Sunny']
}

df = pd.DataFrame(weather_data)
print(df)
print(df['day'])
print(df[['day','temperature']])
type(df['day'])
df['temperature'].max()

df[df['temperature']>32]

df['day'][df['temperature'] == df['temperature'].max()] # Kinda doing SQL in pandas

df[df['temperature'] == df['temperature'].max()] # Kinda doing SQL in pandas

df['temperature'].std()

df['event'].max() # But mean() won't work since data type is string

df.describe()

df.set_index('day',inplace=True)
print(df)

df.index

print(df.loc['1/2/2017'])
df.reset_index(inplace=True)
df.set_index('event',inplace=True)
print(df.loc['Sunny'])


