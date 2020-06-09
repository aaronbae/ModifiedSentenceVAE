import sys
import pandas as pd
data = pd.read_csv("data/qqp_test.csv")
with open("data/qqp_test.txt", 'w', encoding="utf-8") as file:
    for i in range(len(data)):
        file.write(str(data.iloc[i].question1)+"\n")
        file.write(str(data.iloc[i].question2)+"\n")
    file.close()
print('First is done')
    
data = pd.read_csv("data/qqp_train.csv")
with open("data/qqp_train.txt", 'w', encoding="utf-8") as file:
    for i in range(len(data)):
        file.write(str(data.iloc[i].question1)+"\n")
        file.write(str(data.iloc[i].question2)+"\n")
    file.close()
print('all finished')
