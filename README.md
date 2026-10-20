https://gerardfp.github.io/galton_llm/

> una web con una simulación quasi-física de una maquina tipo-galton que en lugar de clavos o pivotes tiene knobs. la máquina sirve como analogía primitiva a un llm. En la parte de arriba caen unas bolas de distintos tamaños que representan palabras. En la parte de abajo están todas las palabras posibles (unas 20 o 30)

> ahora debe haber un sistema de backpropagation, en el que se puede decir a la máquina cual era la palabra que tendría que haber salido y entonces reajuste ligeramente los knobs para que pase (también lo puede hacer el usuario a mano)

> Una palabra no debe ser todo bolas del mismo tamaño, sino una combinación de diferentes bolas. (Cada bola representa un número decimal) Debe haber bolas de 6 tamaños diferentes (rojo=pequeña, violeta=grande) Ahora mismo la bola amarilla es demasiado grande y a veces se queda encasquillada entre los knobs

> quizá para que tenga más sentido las bolas no deberían caer todas del centro. Haremos en la parte superior via con dos carriles ligeramente oblicuos que cada vez se van separando más, de forma que las bola pequeñitas iran cayendo antes que las grandes. de esta forma todos los knobs tendrán influencia, y no necesitamos un único knob en la primera fila


---

> ahora no toques nada. creo que la analogía y la idea se entiende perfectamente.solo necesito saber una cosa. pero no la implementes solo explícamela: si ahora entreno el model (ajusto los knobs) para que la siguiente palabra de "el gato duerme en" sea "la", podría reajustar los knobs que la siguiente palabra después de "el gato duerme en la" sea "alfombra", manteniendo que la siguiente palabra de "el gato duerme en" sea "la"?


**Tal como está programada ahora mismo la máquina: NO**, al entrenar la segunda frase borrarías casi por completo lo aprendido en la primera (lo que en redes neuronales se llama **olvido catastrófico** o *catastrophic forgetting*).

Hay **dos motivos** muy concretos por los que ocurre esto en nuestra máquina actual:

### 1. Las bolas de ambas frases son casi las mismas y caen de forma independiente
* **`"el gato duerme en"`** lanza **24 bolas** (las 4 palabras $\times$ 6 bolas).
* **`"el gato duerme en la"`** lanza **esas mismas 24 bolas** exactamente por los mismos sitios del carril + solo **6 bolas nuevas** de la palabra `"la"`.
* Como cada bola cae de forma independiente y al final gana la cubeta que acumula más bolas ([`updatePredictionUI`](file:///home/gerard/galton_llm/app.js#L912-L964)), existe una contradicción matemática si las bolas no cambian de trayectoria al añadir `"la"`:
  * Si los knobs desvían las primeras 24 bolas hacia la cubeta `"la"`, cuando lances `"el gato duerme en la"`, esas 24 bolas **seguirán cayendo en `"la"`** y las 6 bolas extra de `"la"` nunca podrán superar una ventaja de 24 a 6 para que gane `"alfombra"`.

### 2. El Backpropagation actual crea un "embudo global" hacia la última palabra objetivo
* En [`performBackpropStep`](file:///home/gerard/galton_llm/app.js#L661-L745) y [`computeIdealKnobWeightForTargetX`](file:///home/gerard/galton_llm/app.js#L507-L515), cuando le pides que entrene hacia `"alfombra"`, reorienta todos los knobs que tocaron las bolas (y su zona de convergencia) para apuntar hacia la cubeta de `"alfombra"`.
* Es decir, estás entrenando con **un solo ejemplo a la vez** (*batch size = 1*), sobrescribiendo los ángulos que antes apuntaban hacia `"la"`.

### ¿Qué hace un LLM real (o qué necesitaría la máquina) para mantener las dos cosas a la vez con los mismos knobs?

Para que un mismo conjunto de 180 knobs pueda responder `"la"` a la primera frase y `"alfombra"` a la segunda sin tocarlos entre medias, hacen falta dos cosas:

1. **Atención / Contexto posicional (que la última palabra desplace por dónde entran las demás)**:
   En un Transformer, el mecanismo de *Self-Attention* mezcla las palabras entre sí antes de pasar por los pesos. En nuestra analogía mecánica, significaría que al añadir `"la"` al final del prompt, el calibre o el punto de caída en [`computeDropXForBall`](file:///home/gerard/galton_llm/app.js#L391-L403) se desplace ligeramente para toda la frase. Así, `"el gato duerme en"` caería por unos canales de knobs y `"el gato duerme en la"` caería por otros knobs vecinos distintos.
2. **Entrenamiento intercalado (*Batch / Multi-prompt Backprop*)**:
   En vez de ajustar los knobs solo para una frase, el Backprop daría pequeños pasos alternando ambas reglas (`"el gato duerme en" → "la"` y `"el gato duerme en la" → "alfombra"`), de modo que los knobs por los que pasa la primera frase converjan hacia `"la"` y los knobs por los que pasa la segunda converjan hacia `"alfombra"`.
   
  
---

> dime si estoy en lo cierto, o si mi intución va bien encaminada: creo que cuando caen las bolas deberían mover un poco muy poco los knobs (más o menos dependiendo del tamaño de la bola), luego en el paso de backprop, y para el siguiente lanzamiento los knobs deben voler al estado que han quedado desúes del backprop


**Tu intuición es brillantísima y da en el clavo al 100%**, tanto desde el punto de vista mecánico como de lo que ocurre realmente dentro de un LLM.

De hecho, acabas de describir la diferencia fundamental que existe en una red neuronal entre **los Pesos ($W$)** y **las Activaciones / Estado Oculto ($h_t$ o *KV-Cache*)**:

### 1. Durante la caída de una frase (*Forward Pass*): Memoria a corto plazo
* Si antes de tirar las bolas los knobs están en su **ángulo base aprendido ($W_{\text{base}}$)**, y cada vez que una bola choca contra un knob **lo mueve un poquito** por el propio golpe (una bola grande/violeta tiene más masa y lo desvía un poco más que una pequeña/roja):
  * Las primeras bolas de la frase (`"el"`, `"gato"`, `"duerme"`) van dejando un **rastro físico temporal** en los knobs que tocaron.
  * Cuando caen detrás las bolas de `"en"` o de `"la"`, se encuentran los knobs ligeramente movidos por las palabras anteriores.
  * **¡Eso hace que el orden y la combinación de bolas interactúen entre sí mecánicamente!** Ya no son bolas independientes: el paso de las primeras condiciona por dónde caen las siguientes (exactamente igual que el *estado oculto* de una RNN o el *KV-Cache* de atención en un Transformer).

### 2. En el Backpropagation y en el siguiente lanzamiento: Pesos base ($W_{\text{base}}$)
* Ese pequeño movimiento causado por el golpe de las bolas es solo una **perturbación temporal de esa tirada** (no es aprendizaje permanente).
* Cuando ejecutas el paso de **Backpropagation**, el algoritmo corrige el **estado base ($W_{\text{base}}$)** de los knobs según el error respecto a la palabra objetivo $y^*$.
* Y cada vez que lanzas una frase nueva (o tras el Backprop), los knobs **recuperan exactamente el estado base en el que quedaron tras el último Backprop**, limpios del "empujón temporal" de las bolas del turno anterior.  

¿Quieres que lo implementemos así en la máquina (que cada knob tenga su peso base post-Backprop, que el choque de cada bola lo desvíe un poquito según su masa/tamaño durante la tirada, y que al lanzar de nuevo vuelva a su estado post-Backprop)?
