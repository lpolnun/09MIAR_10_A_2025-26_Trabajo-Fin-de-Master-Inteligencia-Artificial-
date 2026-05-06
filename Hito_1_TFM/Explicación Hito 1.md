**De qué va esto y por qué me parece un problema que merece la pena**

Cuando alguien abre una cuenta bancaria o verifica su identidad desde el móvil, lo habitual es que el sistema le pida una foto de su DNI o pasaporte. Lo que no siempre está claro es si lo que aparece en esa imagen es el documento físico real o alguien intentando engañar al sistema con una impresión en papel, una foto de pantalla o un documento manipulado digitalmente. Ese es exactamente el problema que aborda este TFM: entrenar modelos que sean capaces de distinguir entre una presentación genuina y un ataque de presentación, en el contexto concreto de documentos de identidad en procesos de onboarding digital.

Lo que me parece interesante de este problema, más allá de su aplicación directa en banca o administración pública, es que no tiene una solución obvia. Un ataque bien hecho puede ser difícil de detectar a simple vista, y los sistemas automáticos tienen que aprender a encontrar señales que no siempre son evidentes: la textura de una impresión, los artefactos de una pantalla, inconsistencias de iluminación. Y lo que todavía no está bien resuelto en la literatura es si una red convolucional clásica, que opera de forma local sobre la imagen, tiene ventaja sobre un transformer que captura relaciones globales entre partes de la imagen, o si es al revés. Esa es la pregunta central de este trabajo.

Para situarlo en contexto, la norma internacional que regula cómo se evalúan estos sistemas es la ISO/IEC 30107-3, que define las métricas de APCER y BPCER y establece el protocolo de evaluación. El challenge IJCB PAD-ID Card 2025 usa exactamente ese protocolo, lo que me permite comparar resultados con otros equipos en condiciones controladas.



**Los datos con los que trabajo y las decisiones que tomé antes de tocar una sola línea de código**

Trabajo con dos datasets del challenge IJCB PAD-ID Card. El primero es ID Cards, que combina documentos genuinos con tres tipos de ataque: impresiones en papel, capturas de pantalla y composiciones digitales. El segundo es SYN ID PASS, un conjunto sintético generado con CycleGAN que cubre documentos de Polonia, Portugal y España. Desde el principio decidí mantenerlos completamente separados: el modelo se entrena solo con ID Cards y dejo SYN ID PASS  como test de generalización. La razón de esto es que si llegase a mezclar los dos durante el entrenamiento, el modelo vería la distribución sintética de esos tres países y cuando luego lo evaluase sobre esa misma distribución, las métricas de generalización no me dirían nada real. Así, cuando evalúo en SYN ID PASS, sé que estoy midiendo transferencia genuina a documentos no vistos, que es lo que realmente importa en un sistema de onboarding real.



Planteé el problema como clasificación binaria: genuino frente a ataque, sin diferenciar de entrada entre tipos. Esto está alineado con la norma ISO/IEC 30107-3 y con lo que haría un sistema real, donde la decisión es aceptar o rechazar, no catalogar el método de fraude. Aun así, guardé la etiqueta original de cada muestra (print, screen, composite ) porque más adelante quiero ver si el modelo se comporta distinto ante cada tipo de ataque. Esa comparación es precisamente una de las hipótesis que quiero contrastar: si ViT detecta mejor los screen attacks, donde los artefactos se distribuyen por toda la imagen, mientras ResNet lo hace mejor con los print attacks, donde las señales son más locales.

Un problema que tuve que tener presente desde el principio es el desbalance. El dataset tiene aproximadamente cinco ataques por cada muestra genuina, lo que equivale a una distribución de alrededor del 83% de ataques frente al 17% de genuinos. Si no se hace nada al respecto, el modelo aprende que predecir siempre "ataque" le da un accuracy del 83% y ahí se queda, sin aprender nada útil. Lo gestiono con pesos de clase, pero el desbalance condiciona todo: el entrenamiento, la elección del umbral de decisión y la interpretación de las métricas. No es un detalle menor.



**Cómo lo monté todo y por qué de esa forma**

Trabajo en Google Colab porque SAM y ViT-Base son modelos pesados y en CPU sería inviable. Para los experimentos más serios del pipeline terminé usando una A100, que es la que realmente hace falta cuando SAM procesa las 26.000 imágenes del dataset. Empecé con una T4 y el preprocesado tardaba entre 60 y 90 minutos; con la A100 el tiempo se reduce de forma significativa y el entrenamiento de ViT, que en T4 tenía problemas de memoria con ciertos tamaños de batch, fluye sin problemas.

Lo primero que hice fue sacar los datasets de Drive y extraerlos al almacenamiento local de Colab. Drive tiene mucha latencia de I/O y con 26.000 imágenes eso se nota en cada época: tiempos de carga por batch que deberían ser despreciables se convierten en el cuello de botella real del entrenamiento. Extraer todo a /content/data\_tfm/ al inicio de la sesión y trabajar desde ahí soluciona el problema. Añadí una comprobación antes de descomprimir porque Colab tiene la mala costumbre de reconectarse sin reiniciarse del todo en sesiones largas, y no tiene sentido volver a extraer los ZIPs si ya están en disco. También limpié al principio los artefactos que vienen dentro de los ZIPs generados en Mac (las carpetas \_\_MACOSX y los archivos .DS\_Store), que no aportan nada al dataset pero generan ruido en los recorridos de directorio del DataLoader si no se eliminan.

Fijé las semillas en todas las librerías con aleatoriedad (PyTorch, NumPy, el módulo random de Python) y desactivé el comportamiento no determinista de cuDNN. No mejora el rendimiento del modelo, de hecho lo hace algo más lento, pero me permite comparar experimentos con confianza: si una métrica cambia entre ejecuciones, sé que viene de un cambio real en el pipeline y no de variación aleatoria del entorno. En un trabajo donde voy a estar comparando arquitecturas y configuraciones distintas, eso importa.

Fijé también versiones explícitas de transformers==4.40.0 y peft==0.10.0 en lugar de instalar las últimas disponibles. La razón es porque al combinar LoRA con ViT ambas librerías están muy acopladas, y en pruebas iniciales con versiones más recientes la API de LoraConfig había cambiado lo suficiente como para romper sin ningún aviso claro. Versionar las dependencias es algo que en proyectos pequeños parece burocracia, pero cuando el entorno se puede reiniciar en cualquier momento y la reproducibilidad importa, es la diferencia entre poder retomar el trabajo y tener que depurar media tarde.



L**a exploración del dataset y el etiquetado**

Antes de construir nada me dediqué a entender los datos. Organicé la estructura de carpetas, conté el número de imágenes por categoría y visualicé muestras representativas de cada clase. No es un paso glamuroso, pero es el que evita que aparezca a mitad de un entrenamiento de hora y media un error por una extensión inesperada o una carpeta vacía. En esta fase también confirmé que el etiquetado era consistente con el contenido real de las imágenes que lo que está en la carpeta "screen", son capturas de pantalla y detecté ya el desbalance que mencioné antes.

A partir de esa exploración construí el DataFrame que alimenta todo el resto del pipeline: una fila por imagen, con la ruta, la etiqueta binaria y el tipo original. El campo tipo es el que me permite calcular el APCER desglosado por PAI species en la evaluación, que es uno de los análisis más interesantes para comparar arquitecturas.



**La partición del dataset y por qué no se puede hacer de cualquier manera**

Aquí hay una decisión que me parece importante explicar porque no es obvia. La partición la hago por document\_id, no por imagen. Si dividiera aleatoriamente por imagen, fotos del mismo DNI podrían acabar tanto en entrenamiento como en validación. El modelo habría visto durante el entrenamiento el mismo documento que luego evalúa, lo que infla artificialmente las métricas sin decir nada sobre la capacidad real de generalización. Un 98% de accuracy que viene de haber visto el documento en training no es un buen modelo, es un modelo que memoriza.

Para garantizar que cada documento aparece en un único conjunto uso GroupShuffleSplit de scikit-learn, que agrupa las muestras por document\_id antes de hacer la división. Esto lo vi referenciado en varios trabajos de PAD como práctica estándar para evitar este tipo de data leakage. por ejemplo, en el protocolo del propio IJCB challenge se especifica explícitamente que la partición debe hacerse a nivel de identidad, no de imagen y tiene todo el sentido porque un sistema real se va a encontrar con documentos que nunca ha visto antes, así que lo que hay que medir es exactamente eso.



**La segmentación con SAM y las vueltas que le di**

En los escenarios reales de onboarding, la imagen suele incluir bastante fondo alrededor del documento. Si entreno directamente sobre esas imágenes completas, el modelo puede estar aprendiendo señales del contexto como el color de la mesa, la iluminación de la habitación, el borde del marco... en lugar de las del documento en sí. Eso sería un problema serio: un modelo que detecta ataques porque los ataques suelen fotografiarse sobre fondos distintos no es un modelo que detecta ataques de verdad.

Para resolverlo usé SAM, el modelo de segmentación de Meta (facebook/sam-vit-base). La estrategia que planteé fue la de lanzar un point prompt en el centro de la imagen, que es donde suele estar el documento en este tipo de capturas, y me quedé con la máscara de mayor IoU score de las tres que genera SAM. A partir de esa máscara extraje el bounding box y recorté con un margen de ocho píxeles. Añadí un fallback para los casos en que SAM falla: si la máscara cubre menos del 5% o más del 90% de la imagen, asumo que ha segmentado el fondo o prácticamente todo el frame y aplico un recorte fijo del 8% por borde, que es conservador pero al menos elimina parte del fondo sin riesgo de recortar el documento.

El preprocesado lo hice completamente offline, generé y guardé todas las imágenes segmentadas antes de arrancar el entrenamiento. Si lo hiciera en caliente durante cada época, el coste de SAM se multiplicaría por el número de épocas y el entrenamiento sería absolutamente inviable. Así lo hago una vez, guardo los resultados en disco local y en Drive como backup, y el DataLoader lee siempre desde disco local sin tocar SAM. La función también detecta las imágenes ya procesadas y las salta automáticamente, lo que permite retomar el preprocesado si Colab se reconecta a mitad sin tener que empezar desde cero.

Los resultados de la segmentación fueron bastante buenos. En ID Cards la tasa de detección SAM correcta, es decir, sin caer en el fallback, fue alta, lo que indica que el point prompt central funciona bien para este tipo de capturas donde el documento ocupa una porción grande del frame. En SYN ID PASS los resultados fueron similares. Los casos de fallback correspondían principalmente a imágenes donde el documento estaba muy desplazado del centro o donde el fondo tenía colores muy similares al documento, lo cual es un escenario que SAM con prompt central no maneja bien y que habría que abordar con detección previa del documento en iteraciones futuras.



**Los aumentos de datos y el razonamiento detrás de cada una**

Fijé la resolución de entrada en 224×224 píxeles. Esta elección la condiciona ViT-Base/16, cuyo codificador de posición está aprendido para exactamente 196 parches de 16×16 píxeles. Usar otra resolución obliga a interpolar los embeddings posicionales y eso degrada el encoder. ResNet-50 al ser totalmente convolucional acepta cualquier resolución, pero lo unifiqué en 224 px para poder comparar ambas arquitecturas en igualdad de condiciones. Si uso resoluciones distintas para cada modelo, cualquier diferencia en métricas puede venir de la resolución y no de la arquitectura en sí.

Para los aumentos de datos de entrenamiento intenté simular las condiciones reales de onboarding. Un RandomResizedCrop con escala entre 0.9 y 1.0 introduce variaciones leves de encuadre sin recortar partes importantes del documento. Un RandomHorizontalFlip con probabilidad 0.2 simula que el usuario ha girado ligeramente el teléfono. Una RandomRotation de +-10 grados con fondo blanco y no negro, porque el fondo negro crea un artefacto visual que no aparece en capturas reales y simula inclinaciones de captura. Una RandomPerspective con distorsión 0.1 captura ángulos oblicuos leves. La compresión JPEG aleatoria con calidad entre 40 y 95 simula los artefactos de compresión que introduce el propio teléfono antes de subir la imagen. Un ColorJitter leve cubre variaciones de iluminación, y un GaussianBlur con probabilidad 0.3 simula desenfoque por movimiento.

Ajusté todos estos parámetros. Una rotación excesiva o un flip vertical destruirían la zona MRZ del documento, que es una de las regiones más discriminativas para detectar manipulaciones, si el modelo aprende a usarla, no quiero que los aumentos la eliminen del campo visual. La idea era que cualquier imagen aumentada siguiera pareciendo una captura razonablemente real de un documento, no una imagen distorsionada artificialmente. En validación y test no apliqué ninguna aumentación, solo redimensionado a 256 px seguido de un crop central a 224 px, que es el protocolo estándar para evaluación con modelos preentrenados en ImageNet.



**Por qué ResNet-50 y ViT-LoRA, y no otras arquitecturas**

La elección de estas dos arquitecturas no es aleatoria. Responde a la pregunta central del TFM: ¿tiene el mecanismo de atención de un transformer alguna ventaja real sobre los filtros convolucionales para detectar ataques de presentación en documentos?

Los documentos de identidad tienen características muy distintas según el tipo de ataque. Una impresión en papel introduce texturas de trama regulares, puntos de impresión y una colorimetría ligeramente distinta a la del documento original, señales que son esencialmente locales y que una red convolucional puede capturar con sus filtros. Una captura de pantalla o una composición digital, en cambio, puede introducir artefactos mucho más distribuidos como variaciones sutiles de color a lo largo de toda la imagen, degradados de pantalla, inconsistencias de iluminación que afectan a zonas alejadas entre sí. Para detectar ese tipo de señales se necesita algo que capture relaciones entre partes distantes de la imagen, y eso es exactamente lo que hace el mecanismo de atención de ViT. Esta intuición la encontré referenciada en trabajos como el de Fang et al. (2023) sobre detección de ataques en documentos con transformers, y en trabajos de PAD facial como los de Liu et al. (2021) con ViT, donde ya se reportaba que los transformers tienen ventaja en ataques de mayor fidelidad visual.

Elegí ResNet-50 como baseline convolucional porque es la arquitectura más referenciada en trabajos de PAD sobre documentos de identidad, aparece en el protocolo de referencia del propio challenge, lo que me permite comparar directamente con el estado del arte. He et al. (2016) demostraron que los residual connections permiten entrenar redes mucho más profundas sin degradación del gradiente, y ResNet-50 representa un equilibrio razonable entre capacidad representacional y coste computacional para este problema.

Para el transformer elegí ViT-Base/16 con adaptación LoRA en lugar de hacer fine-tuning completo o congelar el encoder totalmente porque ViT-Base/16 tiene 86 millones de parámetros. Con 14.000 imágenes de entrenamiento, hacer fine-tuning completo de ese encoder implica un riesgo serio de sobreajuste y un coste de memoria GPU que en sesiones de Colab es difícil de gestionar. Congelar el encoder totalmente, por otro lado, limita la capacidad del modelo para adaptarse al dominio específico de los documentos de identidad, que es bastante distinto de ImageNet. LoRA, propuesto por Hu et al. (2021), resuelve ese trade-off inyectando matrices de bajo rango en las proyecciones de atención y dejando el resto del encoder congelado. Con rango r=8 y alpha=16 los parámetros entrenables se quedan en menos del 0,7% del total del modelo, lo que hace viable el fine-tuning sobre este dataset sin coste de memoria excesivo y sin sobreajuste.

Los módulos sobre los que apliqué LoRA son las proyecciones query y value del mecanismo de atención. La justificación viene de los propios autores de LoRA, que muestran que estas proyecciones son las más sensibles a la adaptación de dominio, son las que determinan qué información se selecciona y cómo se pondera, mientras que las proyecciones key son más estables y no aportan ganancia incremental significativa al añadirles adapters. El dropout de LoRA lo fijé en 0.1 para añadir algo de regularización dentro de los adapters sin pasarme.

En este Hito 1 congelo el encoder completo de ResNet y dejo activos solo los adapters LoRA en ViT. Esta decisión es deliberada: primero quiero verificar que el pipeline funciona y que las características de ImageNet son suficientemente transferibles antes de descongelar capas. En el Hito 2 voy a descongelar capas progresivamente en ResNet y voy a explorar rangos de LoRA mayores en ViT para medir el impacto en métricas sin sacrificar generalización.



**Los hiperparámetros y por qué esos y no otros**

Para ResNet usé Adam con learning rate 1e-4. Adam con ese learning rate es el punto de partida estándar para fine-tuning de cabezas de clasificación sobre encoders congelados en visión aparece en decenas de trabajos de transfer learning como punto de partida razonable, y en este caso funciona bien porque el encoder no se mueve y lo único que entrena es la cabeza. Para ViT bajé el learning rate a 5e-5 porque el encoder de ViT, aunque esté congelado en los parámetros principales, tiene adapters LoRA que son más sensibles a variaciones del learning rate al estar inicializados cerca de cero. Un learning rate demasiado alto en los adapters puede hacer que divergan antes de que hayan tenido tiempo de aprender nada útil.

En ambos modelos usé ReduceLROnPlateau con factor 0.5 y patience 2. El scheduler reduce el learning rate a la mitad cuando la pérdida de validación no mejora durante dos épocas seguidas. Es una estrategia conservadora pero adecuada para las cinco épocas de este primer hito, donde no hay tiempo para que el scheduler intervenga muchas veces y la idea es simplemente tener un entrenamiento estable.

El tamaño de batch lo fijé en 32. Con una A100 podría usar batches más grandes, pero 32 es el tamaño que garantiza que los gradientes tienen varianza suficiente para que BatchNorm dentro de la cabeza de ResNet funcione correctamente con batches muy pequeños las estadísticas de BatchNorm se vuelven ruidosas y que el DataLoader puede gestionar sin problemas de memoria incluso con las imágenes segmentadas a 224 px.

Entrené cinco épocas en esta fase. No es suficiente para convergencia, pero es suficiente para verificar que el pipeline funciona de principio a fin, que los gradientes fluyen en ambas arquitecturas y que el entrenamiento no colapsa por problemas de memoria o de tamaño de batch. El objetivo de este hito es ese, no conseguir las mejores métricas posibles.



**Las métricas con las que evalúo y por qué esas**

Evalúo con las métricas del protocolo oficial del challenge, que son las que define la norma ISO/IEC 30107-3. APCER mide la fracción de ataques que el modelo clasifica incorrectamente como genuinos es el error más peligroso en un sistema real, porque significa que un atacante ha pasado el filtro. BPCER mide la fracción de genuinos clasificados como ataques, que es el coste operativo de ser demasiado restrictivo: un usuario legítimo rechazado. EER es el punto de la curva ROC donde los dos errores se igualan, y da una idea del rendimiento global del modelo independientemente del umbral de decisión.

La métrica de ranking oficial del challenge es AVRank, definida como 0.2·BPCER@APCER10% + 0.3·BPCER@APCER5% + 0.5·BPCER@APCER1%. El razonamiento detrás de esta ponderación es que en onboarding digital el error más grave es dejar pasar un ataque, no rechazar un genuino. Por eso la mayor ponderación recae en el punto de operación donde APCER es del 1% en ese punto el sistema es muy restrictivo con los ataques y lo que interesa ver es cuántos genuinos pierde para conseguirlo. Esta métrica penaliza más los modelos que, para conseguir bajos APCER, sacrifican muchos genuinos en el camino.

Además de estas métricas globales, calculo el APCER desglosado por PAI species. Esto me da ya desde este Hito 1 una primera señal de qué tipo de ataque detecta mejor cada arquitectura, aunque con solo cinco épocas los resultados son preliminares.

Para visualizar el comportamiento de cada modelo genero matrices de confusión y curvas DET. La curva DET, Detection Error Tradeoff, es más informativa que la ROC en este contexto porque representa directamente APCER frente a BPCER en toda la curva, que es la relación que importa para PAD, mientras que la ROC mezcla positivos y negativos de una forma menos directa para este problema.



**Cómo guardé todo y por qué me importa la estructura de ficheros**

Guardé en Drive las figuras generadas durante el pipeline, muestras del dataset, comparativas SAM, curvas de aprendizaje, matrices de confusión y curvas DET, y los checkpoints de los modelos al terminar el entrenamiento. Puede parecer algo administrativo, pero tiene una razón práctica concreta: Colab puede perder el entorno en cualquier momento, y si no persisto los checkpoints en Drive, pierdo el entrenamiento y tengo que empezar desde cero. Con los checkpoints guardados puedo retomar desde la época donde lo dejé en cualquier sesión posterior.

También guardé las figuras de forma sistemática porque son las que van a la memoria del TFM. Generarlas durante el entrenamiento y guardarlas en Drive me ahorra tener que reproducir el entrenamiento entero solo para tener los gráficos. El script también incluye al final una función de auditoría que verifica que todas las rutas del dataset existen en disco antes de empezar la evaluación, lo que evita errores silenciosos donde el DataLoader salta imágenes sin avisar y las métricas de evaluación se calculan sobre un subconjunto del dataset sin que me dé cuenta.



**Qué espero contrastar en los siguientes hitos**

La hipótesis principal es que ViT detectará mejor los screen attacks, donde los artefactos son globales y distribuidos, y ResNet lo hará mejor con los print attacks, donde las señales son locales y texturales. Si eso se confirma empíricamente, tendría implicaciones prácticas interesantes: en un sistema de producción podría tener sentido combinar ambos modelos en un ensemble que los trate como complementarios en lugar de competidores.

En el Hito 2 voy a descongelar capas del encoder de ResNet de forma progresiva, probablemente empezando por el último bloque residual, y voy a explorar rangos de LoRA mayores en ViT para medir si hay ganancia en métricas sin sobreajuste. También voy a incorporar análisis de gradientes mediante GradCAM para visualizar qué regiones del documento activa cada modelo, que es la forma de entender si las diferencias de rendimiento responden a estrategias de detección genuinamente distintas. Y voy a explorar si añadir el dataset SYN ID PASS al entrenamiento en lugar de usarlo solo como test externo, mejora la generalización o simplemente cambia la distribución de entrenamiento sin ganancia real.



**Estructura del repositorio**

El notebook genera automáticamente las carpetas figuras/, segmentado/ y checkpoints/ dentro del directorio de Drive TFM\_ID\_Cards/. En figuras/ están todas las visualizaciones del pipeline. En segmentado/ está el backup de las imágenes preprocesadas con SAM. En checkpoints/ están los archivos .pth con el estado de los modelos, el optimizador, el historial de entrenamiento y la configuración LoRA, necesarios para retomar el trabajo en el Hito 2.

