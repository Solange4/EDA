def SED(X, Y):
    return sum((i-j)**2 for i, j in zip(X, Y))

def nearest_neighbor_bf(query_points, reference_points):
    return {
        query_p: min(
            reference_points,
            key=lambda X: SED(X, query_p),
        )
        for query_p in query_points
    }

# Funcion para imprimir un arreglo
def imprimir(arr):
    f = open ('Salida.txt','w')
    f.writelines(str(arr))
    f.close() 


# def lectura():
#     arreglo = []
#     arr = []
#     arch = open('./datos/test100.csv', 'r')
#     for linea in arch:
#         arreglo.append(int(linea))
#     arch.close() 
#     for i in range (100):
#         arr.append(arreglo[i])
#     zipped = zip(arr[0::2], arr[1::2])
#     zipped
 
# Prueba
reference_points = [ (1, 2), (3, 2), (4, 1), (3, 5) ]
query_points = [
    (3, 4), (5, 1), (7, 3), (8, 9), (10, 1), (3, 3)
]

nearest_neighbor_bf( query_points = query_points, reference_points = reference_points,)