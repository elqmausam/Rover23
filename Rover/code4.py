n = [[3,8],[8,48],[3,48]]
p = 0
t = 0
while t <= len(n)-1:
 e = len(n[t])
 for i in range(0, e):
    for j in range(1, e):
      c= n[t][i] & n[t][j]

      if c == 0:
        p+=1
 t+=1

print(p)

