# Machine Learning clasificación con SKLearn

### Preparando el ambiente

**¡Hola!**

Bienvenido(a) al curso Introducción a la clasificación con SKlearn. Mi nombre es Álvaro Camacho y en este curso vamos a explorar datos utilizando [Pandas](https://pandas.pydata.org/pandas-docs/stable/ "Pandas"), visualizarlos utilizando [Seaborn](http://seaborn.pydata.org/introduction.html "Seaborn") y hacer modelos de clasificación utilizando [SKlearn](https://scikit-learn.org/stable/ "SKlearn").

**Ambiente de análisis**

En este curso utilizaremos el Google Colaboratory (Colab), que es un ambiente notebook que no necesita ninguna configuración previa de nuestra parte.

**Colab**

Para usar este ambiente, apenas es necesario tener una cuenta Gmail y cada notebook estará almacenado en nuestro Google Drive. En caso de que no tengas una cuenta Gmail, puedes abrir una [aquí](https://accounts.google.com/signup/v2/webcreateaccount?flowName=GlifWebSignIn&flowEntry=SignUp "aquí"). Para entrar al Colab haz clic [aquí](https://colab.research.google.com/ "aquí").

**Información importante sobre el Colab**

El código de nuestro notebook es ejecutado en una máquina virtual vinculada con nuestra cuenta Gmail. Las máquinas virtuales son recicladas cuando cerramos la ventana o cuando dejamos el notebook con mucho tiempo ocioso.

Para restaurar el notebook, puede ser que sea necesario realizar nuevamente upload de nuestro archivo CSV y ejecutar las opciones *Runtime* y *Restart and run all*.

¿Se puede utilizar otro ambiente para este curso? **¡Claro!** Por ejemplo, si quieres utilizar el ambiente Anaconda, visto anteriormente en el curso de pandas, puedes hacerlo.

**¡Empecemos!**

### Haga lo que hicimos
Llegó la hora de poner en práctica todo lo aprendido en esta lección. Es importante que implementes todo lo que fue visto hasta ahora para continuar con la próxima lección (si ya lo has hecho, ¡excelente!). Implementar lo visto hasta ahora te ayudará a seguir aprendiendo y te dejará más preparado para lo que viene en los próximos videos. En caso de que ya domines esta parte, al final de cada lección podrás descargar el proyecto hasta lo último visto en clase.

1. En una celda de código de tu Notebook digita y ejecuta los siguientes comandos. (Ten en cuenta que los valores dados a los atributos de los animales fueron asignados de manera aleatoria. Si lo deseas, puedes cambiar estos valores) Digita y ejecuta:

```python
# features 1 = sí y 0= no
# tiene el pelo largo?
# tiene las uñas afiladas?
# hace miau?

perro1= [0,1,1]
perro2= [1,0,1]
perro3= [1,1,1]
gato1= [0,1,0]
gato2= [0,1,1]
gato3= [1,1,0]

x_train= [perro1, perro2, perro3, gato1, gato2, gato3]
y_train = [1,1,1,0,0,0]
```

2. Importaremos el paquete `LinearSVC` de `SKLearn Support Vector Machines` y lo instanciaremos en una variable llamada modelo. Digita y ejecuta:

```python
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(x_train,y_train)
```

3. Vamos a introducir al modelo entrenado un animal misterioso para que nuestro clasificador pueda estimar si se trata de un perro o de un gato. Digita y ejecuta:
```python
animal_misterioso= [1,1,1]
model.predict([animal_misterioso])
```
¿Qué animal estimó el clasificador según los atributos informados?

4. Crearemos una lista con animales misteriosos y la emplearemos para probar la exactitud de nuestro modelo. Digita y ejecuta:
```python
misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

x_test = [misterio1, misterio2, misterio3]
y_test = [0,1,1]

previsiones= model.predict(x_test)
```

5. Ahora, vamos a calcular la tasa de acierto de nuestro modelo con los siguientes comandos. Digita y ejecuta:
```python
correctos = (previsiones==y_test).sum()
total = len(x_test)
tasa_de_acierto = correctos/total
print(f'La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%')
```
¿Cuál fue la tasa de acierto de tu modelo? (Recuerda que puedes reentrenar tu modelo cuántas veces quieras con los valores de atributos que desees, e incluso añadir más observaciones)

6. Sklearn nos ofrece una manera sencilla de medir la exactitud a través del `accuracy_score`. Lo importaremos y lo emplearemos como se muestra a continuación. Digita y ejecuta:
```python
from sklearn.metrics import accuracy_score

tasa_de_acierto = accuracy_score(y_test, previsiones)
print(f'La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%')
```

Esta vez, ¿Cuál fue el valor de exactitud de tu modelo?

### Lo que aprendimos en el aula

En esta lección aprendimos a:

- Entrenar modelos/algoritmos.
- Definir características (features) de lo que deseamos clasificar.
- Clasificar en categorías.
- Utilizar los módulos **LinearSVC** y **accuracy_score**.
- Utilizar el método *fit*.
- Hacer predicciones con la función *predict*.
- Calcular el porcentaje de acierto del modelo.
- Comparar las predicciones con los datos de test.
- Utilizar la función *sum*.
- Padronizar nombres.

### Proyecto del aula anterior

¿Comenzando en esta etapa? Aquí puedes descargar los archivos del proyecto que hemos avanzado hasta el aula anterior.

[Descargue los archivos en Github](https://github.com/alura-es-cursos/1918-machine-learning-clasificacion-con-sklearn/blob/aula-2/ML_clasificacion_con_SKLearn.ipynb "Descargue los archivos en Github") o haga clic [aquí](https://github.com/alura-es-cursos/1918-machine-learning-clasificacion-con-sklearn/archive/refs/heads/aula-2.zip "aquí") para descargarlos directamente.

### Preparando el ambiente

Para que puedas desarrollar tus ejercicios, aquí te dejo la url del [dataset](https://gist.githubusercontent.com/ahcamachod/38673f75b54ec62ffc290eff8e7c716e/raw/6eaa07e199d9f668bf94a034cb84dac58c82fa4f/tracking.csv "dataset") que utilizaremos durante el aula.

### Haga lo que hicimos

Llegó la hora de poner en práctica todo lo aprendido en esta lección. Es importante que implementes todo lo que fue visto hasta ahora para continuar con la próxima lección (si ya lo has hecho, ¡excelente!). Implementar lo visto hasta ahora te ayudará a seguir aprendiendo y te dejará más preparado para lo que viene en los próximos videos. En caso de que ya domines esta parte, al final de cada lección podrás descargar el proyecto hasta lo último visto en clase.

1. Iniciaremos un nuevo proyecto. Vamos a importar nuestro dataset directamente desde internet. Digita y ejecuta:

```python
import pandas as pd
uri = 'https://gist.githubusercontent.com/ahcamachod/38673f75b54ec62ffc290eff8e7c716e/raw/6eaa07e199d9f668bf94a034cb84dac58c82fa4f/tracking.csv'
datos = pd.read_csv(uri)
datos.sample(5)
```

2. Ahora, vamos a cambiar los nombres de las columnas para que nuestros atributos queden escritos en español. Al final traeremos una muestra del dataset con 3 registros:
```python
mapa = {
          "home":"principal",
          "how_it_works":"como_funciona",
          "contact":"contacto",
          "bought":"compro"
        }

datos = datos.rename(columns=mapa)
datos.sample(3)
```

3. Vamos a separar nuestros atributos de nuestra clasificación. Para ello digita y ejecuta:

```python
x = datos[['principal','como_funciona','contacto']]
y = datos.compro
```

4. Veremos la forma de nuestro dataset completo antes de separar nuestra base de datos para entrenamiento y para pruebas:
```python
datos.shape
```
¿Cuál es la forma del dataset?

5. Separaremos, entonces de forma manual nuestro dataset, y tomaremos los primeros 75 registros para entrenamiento y los últimos 24 para pruebas:
```python
x_train = x[:75]
y_train = y[:75]
x_test = x[75:]
y_test = y[75:]
```

6. Haremos un print para mostrar con cuántos elementos entrenaremos y con cuántos realizaremos nuestras pruebas:
```python
print(f"Entrenaremos con {len(x_train)} elementos y probaremos con {len(x_test)} elementos.")
```

7. Apoyados en el código del aula anterior, vamos a entrenar un modelo Lineal de SVC y lo probaremos utilizando los siguientes comandos:
```python
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
model.fit(x_train,y_train)
previsiones= model.predict(x_test)

tasa_de_acierto = accuracy_score(y_test, previsiones)
print(f'La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%')
```
¿Cuánto obtuviste en la exactitud del modelo?

8. SKlearn nos ofrece una manera sencilla de segmentar nuestros datos de entrenamiento y de prueba utilizando la función `train_test_split`. Ella recibe varios parámetros, pero los más importantes son la cantidad de muestras para realizar nuestras pruebas, y el estado de aleatoriedad para que siempre que ejecutemos la separación de datos de prueba y entrenamiento entonces lo haga de la misma manera. Esto va a garantizar la replicabilidad de nuestros resultados siempre que se ejecute el algoritmo. Digita y ejecuta:

```python
from sklearn.model_selection import train_test_split

SEED=42

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=SEED)

model = LinearSVC()
model.fit(x_train,y_train)
previsiones= model.predict(x_test)

tasa_de_acierto = accuracy_score(y_test, previsiones)
print(f'La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%')
```

9. Por último, vamos a hacer una estratificación para realizar nuestra separación de las bases de entrenamiento y de prueba, de modo que quede la misma proporción de las clases en ambos casos. Por ejemplo, si en mi dataset de entrenamiento el 30% de las clasificaciones pertenecen a la clase = 1, entonces lo ideal es que mi dataset de pruebas tenga también el 30% de sus clasificaciones como clase = 1. Para ello, configuraremos el parámetro `stratify=y` para que haga la estratificación con base en la clasificación. Digita y ejecuta:
```python
from sklearn.model_selection import train_test_split

SEED=42
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=SEED, stratify=y)

model = LinearSVC()
model.fit(x_train,y_train)
previsiones= model.predict(x_test)

tasa_de_acierto = accuracy_score(y_test, previsiones)
print(f'La tasa de acierto fue de: {round(tasa_de_acierto*100,2)}%')
```
¿Cuál sería la tasa de acierto si a la variable `SEED` le reasignamos el valor `99`?

###  Lo que aprendimos en el aula
En esta lección aprendimos a:

- Abrir archivos CSV.
- Imprimir las primeras observaciones con la función *head*.
- Redefinir el nombre de las columnas.
- Utilizar la función *shape* para ver la cantidad de elementos.
- Separar datos en *Train* y *Test*.
- Tener control de la generación de números aleatorios.
- Utilizar la función* value_counts*.

### Proyecto del aula anterior

¿Comenzando en esta etapa? Aquí puedes descargar los archivos del proyecto que hemos avanzado hasta el aula anterior.

[Descargue los archivos en Github](https://github.com/alura-es-cursos/1918-machine-learning-clasificacion-con-sklearn/blob/aula-3/ML_clasificacion_con_SKLearn.ipynb "Descargue los archivos en Github") o haga clic [aquí](https://github.com/alura-es-cursos/1918-machine-learning-clasificacion-con-sklearn/archive/refs/heads/aula-3.zip "aquí") para descargarlos directamente.

### Preparando el ambiente

Para que puedas desarrollar tus ejercicios, aquí te dejo la url del [dataset](https://gist.githubusercontent.com/ahcamachod/7c55640f0d65bcbd31bb986bb599180c/raw/1b616e97a8719b3ff245fcdd68eaebdb8da38082/projects.csv "dataset") que utilizaremos durante el aula.