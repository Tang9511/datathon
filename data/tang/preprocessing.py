import pandas as pd


dict1 = {
    'polyu_cs': 'JS3868',
    'polyu_nursing': 'JS3648',
    'polyu_business': 'JS3131',
    'bu_science': 'JS2510',
    'hku_engineering': 'JS6963',
    'hku_science': 'JS6901',
    'hku_bba': 'JS6781',
    'hku_nursing': 'JS6468',
    'cuhk_engineering': 'JS4401',
    'cuhk_science': 'JS4601',
    'cuhk_bba': 'JS4202',
    'cuhk_nursing': 'JS4513'
}

year_array = ['2012', '2013', '2014', '2015', '2016', '2017']



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


def get_mean_var_dict():
    mean_dict = {}
    var_dict = {}



    for year in year_array:
        for key, value in dict1.items():
            mean, var = mean_var("data/" + year + "/" + value + ".csv")
            mean_dict[key + "_" + year] = mean
            var_dict[key + "_" + year] = var
    return mean_dict, var_dict