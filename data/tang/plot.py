import pandas as pd
import matplotlib.pyplot as plt

path='features(v2.4)(1).csv'
file=pd.read_csv(path)
polyu_data=file.loc[file['university']=='polyu']
sub_list=polyu_data['subject'].unique()
x=[]
y=[]
for j in range(len(sub_list)):
    x = []
    y = []
    for i in range(6):
        pointx=2012+i
        pointy=polyu_data.loc[(file['subject'] == sub_list[j])&(file['year'] == pointx), 'total_offer'].values
        x.append(pointx)
        y.append(pointy)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("year")
    plt.ylabel("total_offer")
    plt.title("subject: "+sub_list[j])
    plt.savefig(sub_list[j]+"_total_offer.jpg")