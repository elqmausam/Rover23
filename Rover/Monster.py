N = int(input())
Q = int(input())
A = []
O = []
o1 = []
o2 = []
r1 = []
for i in range(0, N):
    l = list(map(int, input().split()))
    o1.append(l[2])
    A.append(l)
    l = []

B = []
for i in range(0, Q):
    l1 = list(map(int, input().split()))
    o2.append(l1[1])
    B.append(l1)
    l1.clear()
print(A)
for i in range(0, N):
    for j in range(A[i][0],A[i][1]+1):
        O.append(j)
t = len(O) 
h = 0
m = 1
for i in range(0, len(o2)):

    for j in range(0, len(o1)):
        if o2[i] >= o1[j]:
            if h == 0 and m == 1:
              if B[i][h] >= A[j][m] and B[i][h] <= A[j][m]:
                t = t - 1
    
    r1.append(t)
 
print(O)
print(o1)