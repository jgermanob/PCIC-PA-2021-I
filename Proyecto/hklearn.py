import numpy as np
import concurrent.futures
from threading import Thread
from model import Model
from scipy import optimize


# class hklearn:
'''
Clase LogisticRegression es una implementación de Model
'''
class LogisticRegression(Model):
    '''
        Constructor.
            C : Parámetro de regularización, en el desarrollo matemático del documento lambda = 1/C
            n_jobs : número máximo de hilos que pueden entrenar concurrentemente un modelo, únicamente funciona
                     en regresión logística multiclase
            solver :  Optimizador a utilizar para minimizar la función de costo dado el gradiente 
                      (fmincg, newton-cg, lbfgs)
            max_iter : número máximo de iteraciones que puede correr el solver para converger al mínimo
    '''
    def __init__(self, C = 1.0, n_jobs = None, solver = 'fmincg', maxiter = 50):
        self.C = C
        self.n_jobs = n_jobs
        self.solver = solver
        self.all_theta = []
        self.max_iter = maxiter
        self.decoder = {}

    #Probar con otros solvers
    #Esta función se le asigna a un hilo distinto, en este hilo se realizará el entrenamiento de un modelo de regresión
    #correspondiente a una clase vs. todas las demás
    def func(self, thetas_p, max_iter, n, c, X_p, y_p, C):
        initial_theta = np.zeros((n + 1, 1), dtype=np.float64)
        args = [X_p[c], y_p[c], C]
        print('Iter: ', c)
        #theta= optimize.fmin_cg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=max_iter)
        if self.solver == 'fmincg':
            theta= optimize.fmin_cg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
        elif self.solver == 'newton-cg':
            theta= optimize.fmin_ncg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
        elif self.solver == 'lbfgs':
            theta= optimize.fmin_l_bfgs_b(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)[0]
        thetas_p[c] = theta.transpose()

    #Esta función se encarga de asignar uno a todos los ejemplos de la clase c y 0 a todos los demás
    def func2(self, y_p, c, y, X_p, X):
        X_p[c] = X
        y_p[c] = np.array(list(map(lambda x : 1.0 if x == c else 0.0, y)), dtype=np.float64)  

    #Función principal de entrenamiento
    #Entrena el modelo con un a matriz de ejemplos X y sus respectivas etiquetas y
    def fit(self, X, y):
        labels = set(y)#conjunto de clases
        n_labels = len(labels)#número de etiquetas o de clases
        #creamos un encoder para el mapeo de etiquetas 
        # que podrían ser de tipo distinto a 
        #flotante a valores de tipo flotante de 0 ... n_labels
        encoder = dict(zip(labels, np.arange(float(n_labels))))
        #Mapeo inverso al anterior
        self.decoder = dict(zip(np.arange(float(n_labels)), labels))
        #Número de dimensiones o de características
        n = X.shape[1]
        #Número de ejemplos
        m = X.shape[0]
        
        #Agregamos un uno a cada ejemplo para tomar en cuenta el término de bias
        X_aux = np.concatenate((np.ones((m ,1), dtype = np.float64), X), axis=1)
        #Valor inicial del vector de theta para el optimizador
        initial_theta = np.zeros((n + 1, 1), dtype=np.float64)
        #Vector de parámetros del modelo a entrenar
        theta = np.zeros((n + 1, 1), dtype=np.float64)
        #Vector de etiquetas codificadas a valores reales para poder 
        #utilizarlas en el cálculo de costos y gradiente
        y_enc = np.array(list(map(lambda x : encoder[x], y)))
        #Lista de argumentos que se le pasará a las funciones del optimizador
        args = [X_aux, y_enc, self.C]

        #Si es un problema de clasificación multiclase
        if n_labels > 2:
            #Creamos una matriz donde se va a depositar un vector de parámetros por cada modelo
            #correspondiente a cada etiqueta 
            self.all_theta = np.zeros((n_labels, n + 1), dtype=np.float64)
            #Si no se quiere ejecutar en paralelo
            if self.n_jobs is None:
                #Para cada clase
                for c in range(n_labels):
                    #Asignar uno a todos los ejemplos de la clase c y 0 a todos los demás
                    args[1] = np.array(list(map(lambda x : 1.0 if x == c else 0.0, y_enc)), dtype=np.float64)
                    #Utilizamos el optimizador elegido, y le pasamos la funcion de costo, la theta inicial, el gradiente de la función de costo, y la lista de argumentos 
                    #que se le pasará tanto a la función de costo como al gradiente
                    if self.solver == 'fmincg':
                        theta= optimize.fmin_cg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
                    elif self.solver == 'newton-cg':
                        theta= optimize.fmin_ncg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
                    elif self.solver == 'lbfgs':
                        theta= optimize.fmin_l_bfgs_b(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)[0]
                        #print(theta)
                    #Guardamos el vector del modelo entrenado para la clase c
                    #trasponemos el vector columna para colocarlo en su renglón correspondiente
                    #en la matriz de vectores de modelo
                    self.all_theta[c, :] = theta.transpose()
            else:
                #Si se quiere ejecutar en paralelo
                #Diccionario de 'y', para cada clase vamos a modificar 
                #los valores de y para que sean uno para los ejemplos de la clase
                #y cero para todos los demás, si no depositamos en un diccionario
                #estos vectores, varios threads van querer modificar al mismo tiempo
                #los valores de 'y', llevando a corrupción de los datos
                y_p = {}
                #Misma lógica que para los valores de 'y', cada thread va a entrenar
                #y por lo tanto a modificar los valores de theta
                thetas = {}
                X_p = {}
                #Creamos un threadpool para evitar crear y destruir constantemente threads
                #y tan sólo crear un conjunto fijo de threads al inicio de la ejecución
                with concurrent.futures.ThreadPoolExecutor(max_workers = self.n_jobs) as executor:
                    #Vamos a ejecutar el for de manera concurrente con un límite de concurrencia de n_jobs
                    #cada thread ejecuta una iteración del for, cualquier bloque de código que se coloque
                    #dentro del for se ejecuta de manera serial en el contexto del thread que lo esté ejecutando
                    for c in range(n_labels):
                        #Convertimos los valores de 'y' y los asignamos al diccionario y_p en su llave correspondiente
                        future = executor.submit(self.func2, y_p, c, y_enc, X_p, X_aux)
                        #Ejecutamos el entrenamiento del modelo para la clase c
                        future = executor.submit(self.func, thetas, self.max_iter, n, c, X_p, y_p, self.C)
                #Al terminar de entrenar cada modelo, asignamos cada vector de cada modelo a su renglón correspondiente
                #dentro de la matriz de modelos
                for c in range(n_labels):
                    self.all_theta[c,:] = thetas[c]

        #Para la regresión logística binaria
        else:   
            #Tenemos un solo modelo, que nos servirá para determinar los ejemplos que son de una clase
            # y los que no lo son (y evidentemente pertenecen a la otra clase)        
            self.all_theta = np.zeros((1, n + 1), dtype=np.float64)
            if self.solver == 'fmincg':
                theta= optimize.fmin_cg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
            elif self.solver == 'newton-cg':
                theta= optimize.fmin_ncg(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)
            elif self.solver == 'lbfgs':
                theta= optimize.fmin_l_bfgs_b(self.cost_func, initial_theta, fprime = self.grad_cost_func, args = args, maxiter=self.max_iter)[0]
                #print(theta)
            self.all_theta = theta.transpose()

    #El modelo entrenado, predice con base en una entrada X de ejemplos
    def predict(self, X):
        #número de ejemplos es el número de renglones de la matriz de ejemplos
        m = X.shape[0]
        #número de clases o etiquetas es el número de elementos en el mapeo de valores reales a etiquetas
        num_labels = len(self.decoder)
        #Añadimos una columna de unos a la matriz X para tomar en cuenta el término de sesgo
        X_aux = np.concatenate((np.ones((m,1), dtype = np.float64), X), axis=1)
        #Para obtener las predicciones para cada clase, basta con calcular g de la multiplicación de 
        #la matriz de ejemplos X y la traspuesta de la matriz de vectores de modelo
        s = self.sigmoid(np.matmul(X_aux, self.all_theta.transpose()))
        #Creamos el vector de predicciones
        if num_labels > 2:          
            #Para la regresión logística multiclase por cada valor obtenido de g sobre cada ejemplo
            #se obtiene el que sea máximo y con base en el índice, tendremos el valor de la predicción de la 
            #clase a la que pertenece el ejemplo en cuestión, para después, de ese valor real, decodificarlo a su
            #etiqueta original
            return np.array(list(map(lambda x : self.decoder[np.where(x == np.amax(x))[0][0]], s)))
        else:     
            #Para el caso de la regresión logística binaria, como únicamente tenemos un modelo,
            #simplemente diremos que si x>=0.5 pertenece a la clase con el valor real 1 y para x < 5, 
            #pertenece a la clase 0, ambos valores los decodificamos a sus valores de etiqueta original
            return np.array(list(map(lambda x : self.decoder[1. if x >= 0.5 else 0.], s)))


    
    #Función logística, corresponde en el desarrollo matemático a g(z)
    def sigmoid(self, z):
        g = 1./(1. + np.exp(-z, dtype=np.float64))
        return g

    #Función de costo a optimizar en el entrenamiento
    #corresponde a J(theta) en el desarrollo matemático
    def cost_func(self, theta, *args):
        X, y, C = args
        m = X.shape[0]
        theta_aux = theta.copy()
        theta_aux[0] = 0 
        #Función de costo vectorizada conforme al desarrollo matemático
        cost = (1/m)*(np.matmul(-y.transpose(),np.log(self.sigmoid(np.matmul(X,theta, dtype=np.float64))), dtype=np.float64) 
        - np.matmul((1-y).transpose(),np.log(1-self.sigmoid(np.matmul(X,theta, dtype=np.float64))), dtype=np.float64))
        reg_term = (1/(2*C*m))*(np.matmul(theta_aux.transpose(),theta_aux, dtype=np.float64))
        J = cost + reg_term
        return J

    #Gradiente de la funcion de costo necesaria para optimizar la función de costo de manera más eficiente,
    #Corresponde a gradJ(theta) en el desarrollo matemático
    def grad_cost_func(self, theta, *args):
        X, y, C = args
        m = X.shape[0]
        I = np.eye(len(theta))
        I[0,0] = 0
        #Gradiente vectorizado conforme al desarrollo matemático
        grad = (1/m)*np.matmul(X.transpose(), (self.sigmoid(np.matmul(X,theta)) - y), dtype=np.float64)+ (1/(C*m))*(np.matmul(I,theta, dtype=np.float64))
        return grad[:]
