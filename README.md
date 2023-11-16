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