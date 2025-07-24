l1 = [3,8,48]
l3 = []
lists = []
for i in range(0,len(l1)):
        for j in range(i+1, len(l1)):
            l3.append(l1[i])
            l3.append(l1[j])
            lists.append(l3)
            l3.clear()
print(lists)