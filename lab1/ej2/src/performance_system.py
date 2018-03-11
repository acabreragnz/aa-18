import experiment_generator as eg


n = 15     
A = [0,2,3,3,4,4,5,5,2,3,3,4,4,5,5] 
A1 = [1,0,0,0,0,4,5,5,0,0,0,0,4,5,5] 



def convertir(T):
    #[x2,y2,x3,y3,x4,y4,x5,w2,z2,w3,z3,w4,z4,w5]
    return [0,0,0,0,0,0,0,0,0,0,0,0,0];


def V(T, A):
    #return a0 + a1x2 + a2y2 + a3x3 + a4y3 + a5x4 + a6y4 + a7x5 + a8w2 + a9z2 + a10w3 + a11z3 + a12w4 + a13z4 + a14w5
    return A[0] + A[1]*T[0] + A[2]*T[1] + A[3]*T[2] + A[4]*T[3] + A[5]*T[4] + A[6]*T[5] + A[7]*T[6] + A[8]*T[7] + A[9]*T[8] + A[10]*T[9] + A[11]*T[10] + A[12]*T[11] + A[13]*T[11] + A[14]*T[12]



def jugada(T,turn,A):
    #V es una función de evaluación que asigna una puntuación numérica a cualquier estado de tablero.
    #Pretendemos que esta función objetivo V asigne puntuaciones más altas a mejores estados de tablero. 
    #Obener la mejor jugada se puede lograr generando el estado del tablero sucesor producido por cada jugada legal, 
    #luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

    #Determinar todas las jugadas posibles para el turno pasado por parametro - para cada celda vacia de la matriz purebo a colocar una ficha del color
    #del turno que esta jugando
    
    v_max = V(convertir(T),A);    
    T_next = [[T[x][y] for x in range(n)] for y in range(n)] 
    T_result = []
    for i in range (0, n):
        for j in range (0, n):
            if T_next[i][j] == 0 :
                T_next[i][j] = turn
                v_result = V(convertir(T_next),A)
                if v_result >= v_max :
                    v_max = v_result
                    T_result = [[T_next[x][y] for x in range(n)] for y in range(n)] 
                T_next[i][j] = 0
    
    return T_result

#BLACK = 1
#WHITE = 2 

T = eg.experimentGenerator(n)
turn = 1
print(T)
T = jugada(T, turn,A)
print(T)
