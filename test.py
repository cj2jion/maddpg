import numpy as np 
def fun(low,high):
    sum1,sum2=0
    i=low
    while(i<high):
        issprise=1,k=2
        while(k<=np.sqrt(i)):
            if((i%k)==0):
                issprise=0
                k+=1
        if(issprise==1):
            if(i<10):
                sum2+=1
            else:
                sum1+=int(i%100/10)
                sum2+=i%10
        i+=1
    print(min(sum1,sum2))
    return 0

fun(151,160)