"""n = int(input())
l = list(map(int, input().split()))
A = 1000007
l1 = [l[i] for i in range(len(l)) if l.index(l[i])==i]

if len(l1)%2 == 0:
    m = len(l1)
else:
    m = (len(l1)+1)/2

for x in range(1,A):
    if(((A%m)*(x%m))%m==1):
        return x
 return -1
print(modin(A,m))"""
def modInverse(A, M):
 

    for X in range(1, M):

        if (((A % M) * (X % M)) % M == 1):

            return X

    return -1
 
 


if __name__ == "__main__":

    n = int(input())
    l = list(map(int, input().split()))
    M = 1000007
    l1 = [l[i] for i in range(len(l)) if l.index(l[i])==i]

    if len(l1)%2 == 0:
       A = int(len(l1)/2)
    else:
       A = int((len(l1)+1)/2)
 

    
    
    print(modInverse(A, M))