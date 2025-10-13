import pathlib
import math
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    lines = path.read_text().replace(" ","").split("\n")
    A = []
    B = []
    c= ["x", "y", "z"]
    for s in lines:
        x, y = s.split("=")
        B.append(float(y))
        y=x
        l=[]

        for i in c:
           if y.find(i)==-1:
                l.append(0.0)
                continue
           y= y.split(i)
           x=y[0]

           if x=="+"or x=="":
               l.append(1)
           elif x=="-":
               l.append(float(-1.0))
           else:
               l.append(float(x))
           y = y[-1]

        A.append(list(l))

    return A, B
def determinant(matrix: list[list[float]]) -> float:
    return matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])+matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0])
def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0]+matrix[1][1]+matrix[2][2]
def norm(vector: list[float]) -> float:
    return math.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return list(zip(matrix[0],matrix[1],matrix[2]))

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    l=[0,0,0]
    for index,i in enumerate (matrix):
        s=0
        for index_2,j in enumerate (i):
            l[index]+=(vector[index_2]*j)
    return l

def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    x=[]
    y=[]
    z=[]
    for i in matrix:
        x.append(i.copy())
    print(matrix)
    print(x)
    x[0][0]=vector[0]
    x[1][0] = vector[1]
    x[2][0] = vector[2]

    x=determinant(x)/determinant(matrix)

    for i in matrix:
        y.append(i.copy())

    y[0][1]=vector[0]
    y[1][1] = vector[1]
    y[2][1] = vector[2]
    y=determinant(y)/determinant(matrix)

    for i in matrix:
        z.append(i.copy())
    z[0][2]=  vector[0]
    z[1][2] = vector[1]
    z[2][2] = vector[2]
    z=determinant(z)/determinant(matrix)

    return [x, y, z]

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    m=[]
    for i_1, x in enumerate (matrix):
        l=[]
        for j_1,y in enumerate(x):
            if i_1!=i and j_1!=j:
                l.append(y)
        if l:
            m.append(l)
    return m


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    c=[]
    for i, x in enumerate (matrix):
        l=[]
        for j,y in enumerate(x):
                mi=minor(matrix,i,j)
                l.append(((-1)**(i+j))*(mi[0][0]*mi[1][1]-mi[1][0]*mi[0][1]))
        if l:
            c.append(l)
    return c

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    adj=adjoint(matrix)
    det=determinant(matrix)
    inversa=list()
    for i,x in enumerate (adj.copy()):
        l=[]
        for j,y in enumerate(x):
            l.append(adj[i][j]/det)
        inversa.append(l)
    print(f"{inversa}")
    return multiply(inversa,vector)

A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{minor(A,1,1)=}")
print(f"{cofactor(A)=}")
print(f"{adjoint(A)=}")
print(f"{solve(A, B)=}")