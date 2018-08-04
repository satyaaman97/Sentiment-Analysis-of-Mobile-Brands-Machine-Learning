from twitterscraper import query_tweets
import csv
import random
import time
d=list()
e=list()
for tweet in query_tweets("@SamsungMobile", 300)[:300]:
	
	c=tweet.text.encode('utf-8')
	d.append(c)
				
f=open("C:\Python27\data\witter"+"\\"+"test-neg.txt",'w') 
for j in d:
	f.write(j)
	f.write('\n')		

f.close()

for tweet in query_tweets("@SamsungMobileUS", 300)[:300]:
	
	c=tweet.text.encode('utf-8')
	e.append(c)
				
f1=open("C:\Python27\data\witter"+"\\"+"test-pos.txt",'w') 
for j in e:
	f1.write(j)
	f1.write('\n')		

f1.close()