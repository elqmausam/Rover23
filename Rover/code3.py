def sub_lists (l1):
    lists = []
    l3 = []
    for i in range(0,len(l1)):
        for j in range(i+1, len(l1)):
            l3.append(l1[i])
            l3.append(l1[j])
            lists.append(l3)
    return lists

if __name__ == "__main__":
    n = int(input())
    l1 = list(map(int, input().split()))
    l2 = []
    l = sub_lists(l1)
    for i in range(0, len(l)):
        if len(l[i]) > 1:
            l2.append(l[i])
    p = 0
    t = 0
    while t <= len(l2)-1:
     e = len(l2[t])
     for i in range(0, e):
      for j in range(1, e):
       c= l2[t][i] & l2[t][j]

       if c == 0:
        p+=1
     t+=1
    print(l2)
    print(p)




