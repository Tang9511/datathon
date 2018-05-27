import pandas as pd

def mean_var(file):
    content=pd.read_csv(file)
    score=content['Best 5']
    mean=score.mean()
    variance=score.var()
    return mean,variance

def atu(file):
    content=pd.read_csv(file)
    uni=content['university']
    sub=content['subject']
    year=content['year']
    df=pd.concat([uni,sub,year,content['max'],content['min'],content['average']],axis=1)
    return df

file='features(v2.2).csv'
atu(file)