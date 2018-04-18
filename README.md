# aa-18

Tareas del curso Aprendizaje Automático 2018

## Instalar dependencias

1. Instalar virtualenv:

  ```pip install virtualenv```

2. Crear directorio para las dependencias:

  ```virtualenv -p /usr/bin/python3 deps```

3. Activar virtualenv

  ``` source deps/bin/activate```

4. Instalar dependencias:

  ``` pip install -r requirements.txt```


Nota: siempre que se quiera correr python, se debe activar virtualenv

## Jupiter Notebooks

1. Instalar Anaconda:

  - Linux: desde la [página oficial](https://www.anaconda.com/download/#macos)
  - MacOS: desde la [página oficial](https://www.anaconda.com/download/#macos) o con el comando [brew](https://brew.sh/index_es) (**recomendado**)

2. Crear entorno virtual:

  ```conda create -n py36 python=3.6```

3. Activar entorno virtual:

  - MacOS y Linux:

    ```source activate py36```

  En MacOS puede ser necesario situarse en el directorio donde se encuentran los binarios instalados por Anaconda (si fue instalado con brew, por defecto se encuentra en /usr/local/anaconda3/bin).

4. Instalar dependencias:

  **Se deben instalar las mismas dependencias que se usan en todo el proyecto:**

  ```pip install -r <directorio del proyecto>/requirements.txt```

  El directorio del proyecto generalmente se encuentra bajo la ruta del directorio del usuario. En MacOS es /Users/ y en Linux /home .

5. Ejecutar la interfaz gráfica de Anaconda y abrir Jupiter Notebooks para el entorno creado antes (py36).

Notas:

  - Si no está agregada la ruta de los binarios de Anaconda al PATH del sistema, se puede agregar dicha ruta o también ejecutarlos directamente desde el directorio. Por ejemplo en MacOS, si fue instalado con brew, se debe hacer lo siguiente para ejecutar conda:

    1. Ir al directorio /usr/local/anaconda3/bin/

    2. Ejecutar conda de la siguiente forma: ```./conda```

  - Aveces el código no se actualiza bien dentro de un notebook por lo que es necesario reiniciar el kernel.

Mas info:

- Página de Anaconda: https://www.anaconda.com/download/#macos
- Página de brew: https://brew.sh/index_es
- Instalación de conda en Linux: https://conda.io/docs/user-guide/install/linux.html#install-linux-silent
- Manejo de entornos con conda: https://conda.io/docs/user-guide/tasks/manage-environments.html
