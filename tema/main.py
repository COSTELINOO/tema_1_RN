import pathlib
import math


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    lines = path.read_text().replace(" ", "").split("\n")
    a = []
    b = []
    c = ["x", "y", "z"]
    for s in lines:
        x, y = s.split("=")
        b.append(float(y))
        y = x
        lista = []

        for i in c:
            if y.find(i) == -1:
                lista.append(0.0)
                continue
            y = y.split(i)
            x = y[0]

            if x == "+" or x == "":
                lista.append(1)
            elif x == "-":
                lista.append(float(-1.0))
            else:
                lista.append(float(x))
            y = y[-1]

        a.append(list(lista))

    return a, b


def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 1:
        return matrix[0][0]

    line = matrix[0].copy()
    s = 0
    for i, x in enumerate(line):
        s = s + (((-1) ** (1 + i + 1)) * x * determinant(minor(matrix, 0, i)))
    return s


def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def norm(vector: list[float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(matrix[0], matrix[1], matrix[2])]


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    lista = [0, 0, 0]
    for index, i in enumerate(matrix):
        for index_2, j in enumerate(i):
            lista[index] += (vector[index_2] * j)
    return lista


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    x = []
    y = []
    z = []
    det = determinant(matrix)
    for i in matrix:
        x.append(i.copy())
    x[0][0] = vector[0]
    x[1][0] = vector[1]
    x[2][0] = vector[2]
    x = determinant(x) / det

    for i in matrix:
        y.append(i.copy())

    y[0][1] = vector[0]
    y[1][1] = vector[1]
    y[2][1] = vector[2]
    y = determinant(y) / det

    for i in matrix:
        z.append(i.copy())
    z[0][2] = vector[0]
    z[1][2] = vector[1]
    z[2][2] = vector[2]
    z = determinant(z) / det

    return [x, y, z]


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    m = []
    for i_1, x in enumerate(matrix):
        lista = []
        for j_1, y in enumerate(x):
            if i_1 != i and j_1 != j:
                lista.append(y)
        if lista:
            m.append(lista)
    return m


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    c = []
    for i, x in enumerate(matrix):
        lista = []
        for j, y in enumerate(x):
            mi = minor(matrix, i, j)
            lista.append(((-1) ** (i + j)) * (determinant(mi)))
        if lista:
            c.append(lista)
    return c


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    adj = adjoint(matrix)
    det = determinant(matrix)
    if det == 0:
        print("Nu se poate rezolva, det=0")
        return []
    inversa = list()
    for i, x in enumerate(adj.copy()):
        lista = []
        for j, y in enumerate(x):
            lista.append(adj[i][j] / det)
        inversa.append(lista)
    return multiply(inversa, vector)


A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{minor(A, 1, 1)=}")
print(f"{cofactor(A)=}")
print(f"{adjoint(A)=}")
print(f"{solve(A, B)=}")
