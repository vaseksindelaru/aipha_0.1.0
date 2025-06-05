# Documentación del Sistema de Continuación Inteligente Post-Triple Coincidencia (CIPTC)

Este documento describe la arquitectura y los componentes iniciales del Sistema de Continuación Inteligente Post-Triple Coincidencia (CIPTC). El CIPTC es un sistema diseñado para, tras recibir una señal válida del "Sistema de Triple Coincidencia" (STC), utilizar técnicas avanzadas de Inteligencia Artificial, incluyendo Modelos de Lenguaje Grandes (LLMs), Recuperación Aumentada por Generación (RAG) y modelos predictivos, para analizar el contexto del mercado, proponer y evaluar múltiples escenarios de continuación del precio, y tomar una decisión táctica informada. El diseño se inspira en la filosofía y arquitectura de sistemas agénticos y con integración de conocimiento del dominio, similar a Archon.

## Objetivos del CIPTC

El objetivo principal del sistema CIPTC es ir más allá de la simple detección de señales del STC y añadir una capa de inteligencia para tomar decisiones sobre la acción más probable o rentable a seguir. Específicamente, busca:

1.  **Análisis de Contexto:** Analizar el contexto actual del mercado y la naturaleza específica de la señal STC recibida.
2.  **Propuesta de Continuaciones:** Proponer y evaluar *múltiples* posibles continuaciones del precio tras la señal (ej. ruptura del nivel clave, rebote desde el nivel, falsa ruptura, período de espera).
3.  **Ingeniería de Features:** Identificar y extraer *features* adicionales relevantes para cada continuación propuesta, más allá de los proporcionados por el STC.
4.  **Selección de Modelos:** Seleccionar el modelo predictivo (regresión, clasificación, etc.) más apropiado para predecir el resultado de cada continuación propuesta.
5.  **Predicción:** Entrenar (o cargar) y ejecutar modelos predictivos para estimar la probabilidad o el resultado esperado de cada continuación.
6.  **Toma de Decisión Táctica:** Sintetizar el contexto, las predicciones y el conocimiento del dominio para tomar una decisión táctica sobre la acción a seguir.
7.  **Auto-Mejora (Futuro):** Crear un ciclo de retroalimentación donde los resultados de las decisiones pasadas informen y mejoren futuras predicciones y decisiones.

## Inspiración en Arquitecturas Agénticas (Archon)

El diseño del CIPTC se inspira en principios de arquitecturas agénticas avanzadas, como Archon, adoptando los siguientes conceptos:

*   **Razonamiento Agéntico:** El sistema actúa como un agente autónomo que percibe (la señal STC y contexto), planea (evaluando continuaciones), actúa (tomando una decisión) y evalúa (registrando resultados).
*   **Integración de Conocimiento del Dominio:** Utiliza un sistema RAG robusto para acceder y utilizar información relevante sobre trading, patrones de mercado, documentación de los sistemas subyacentes (STC), y resultados históricos.
*   **Arquitectura Modular:** Se diseña con componentes claros y bien definidos que interactúan entre sí, facilitando el desarrollo, la prueba y la escalabilidad.
*   **Feedback y Refinamiento:** El sistema incorpora mecanismos para aprender de sus resultados pasados y refinar sus procesos de toma de decisión y predicción.
*   **Uso de Herramientas Modernas:** Se contempla el uso de herramientas como Pydantic para la estructuración de datos y LangGraph o herramientas similares para la orquestación de flujos de trabajo complejos.

## Diseño del Sistema CIPTC (Versión Inicial - Alfa 0.1)

### Arquitectura General

La arquitectura inicial se concibe como una serie de componentes interconectados, orquestados por un módulo central.

```
+-----------------------------+      +---------------------------+      +---------------------------+
| Sistema Triple Coincidencia | ---> | CIPTC Orchestrator        | ---> | Execution Engine          |
| (Genera Señal STC)          |      | (LangGraph/Python Script) |      | (Trading API/Simulator)   |
+-----------------------------+      +---------------------------+      +---------------------------+
          |                                     |
          |--------------------------------------|
          |                                     V
+-----------------------------+      +---------------------------+      +---------------------------+
| Aia4you-Core RAG            | <--> | CIPTC Agent(s) / Nodes    | <--> | Predictive Model Hub      |
| (Base de Conocimiento)      |      | (LLM Interaction, Logic)  |      | (Entrenamiento/Predicción)|
+-----------------------------+      +---------------------------+      +---------------------------+
          |                                     ^
          |                                     |
          +-------------------------------------+
          |
+-----------------------------+
| Feature Engineering Module  |
| (Propone y Extrae Features) |
+-----------------------------+
```

**Componentes Detallados:**

1.  **Entrada:** Señal del Sistema de Triple Coincidencia (STC) con su puntuación y datos asociados (vela clave, zona, tendencia).
2.  **Aia4you-Core RAG (Base de Conocimiento):**
    *   **Contenido:** Documentación STC, documentación Archon, conceptos de trading (ruptura, rebote, falsas rupturas, tipos de órdenes), documentación de modelos predictivos (Gradient Boosting, LSTM, Redes Neuronales Simples, Regresión Logística, etc.), análisis históricos post-STC, *resultados y logs de ejecuciones anteriores de CIPTC*.
    *   **Tecnología:** Base de datos vectorial (Supabase/Pinecone), modelo de embedding, interfaz de consulta.
3.  **CIPTC Orchestrator:**
    *   **Función:** Coordina el flujo de trabajo tras recibir una señal STC. Podría ser un script Python complejo o, idealmente, un grafo de LangGraph.
    *   **Pasos Orquestados:**
        *   Recibir señal STC.
        *   Invocar al `Context Analyzer Agent`.
        *   Invocar al `Continuation Proposer Agent`.
        *   Para cada continuación propuesta:
            *   Invocar al `Feature Suggester Agent`.
            *   Invocar al `Model Selector Agent`.
            *   Invocar al `Predictive Model Hub` para entrenamiento/predicción.
        *   Invocar al `Tactical Decision Agent`.
        *   Enviar la decisión final al `Execution Engine`.
        *   Invocar al `Performance Logger Agent`.
4.  **CIPTC Agents / Nodes (Interactúan con LLM y RAG):**
    *   **`Context Analyzer Agent`:**
        *   **Input:** Señal STC.
        *   **Proceso:** Consulta el RAG para obtener contexto relevante sobre la señal específica, condiciones actuales del mercado (si hay datos disponibles), y rendimiento histórico de STC en situaciones similares.
        *   **Output:** Resumen contextual estructurado (Pydantic).
    *   **`Continuation Proposer Agent`:**
        *   **Input:** Señal STC, Resumen Contextual.
        *   **Proceso:** Llama a un LLM (Gemini/Claude) con un prompt: *"Dada esta señal STC [detalles] y el contexto [resumen], propone 2-3 posibles continuaciones de trading realistas (ej. 'Range Breakout', 'Mean Reversion Bounce', 'False Breakout Fade', 'Wait and See'). Justifica brevemente cada una."*
        *   **Output:** Lista de `ProposedContinuation` (objeto Pydantic con nombre y justificación).
    *   **`Feature Suggester Agent`:**
        *   **Input:** Señal STC, Resumen Contextual, `ProposedContinuation`.
        *   **Proceso:** Llama a un LLM: *"Para la continuación '[Nombre Continuación]' en este contexto [resumen], sugiere 3-5 *features* adicionales (más allá de los básicos del STC) que serían útiles para predecir su éxito. Considera indicadores técnicos, patrones de volumen, microestructura de mercado si es posible. Describe por qué cada feature es relevante."*
        *   **Output:** Lista de `SuggestedFeature` (objeto Pydantic con nombre, descripción, fuente de datos potencial). **¡Aquí está mi propuesta de features para cada continuación!**
    *   **`Model Selector Agent`:**
        *   **Input:** Señal STC, Resumen Contextual, `ProposedContinuation`, Lista de `SuggestedFeature`.
        *   **Proceso:** Llama a un LLM: *"Considerando la continuación '[Nombre Continuación]', los features disponibles/sugeridos, y el contexto [resumen], ¿qué tipo de modelo predictivo (ej. Gradient Boosting, LSTM, Logistic Regression, simple Neural Network) sería más apropiado para esta tarea? Justifica brevemente."*
        *   **Output:** `SelectedModelType` (objeto Pydantic con tipo de modelo y justificación). **¡Aquí elijo el modelo!**
    *   **`Tactical Decision Agent`:**
        *   **Input:** Señal STC, Resumen Contextual, Predicciones de todos los modelos evaluados (probabilidad, rentabilidad esperada, confianza).
        *   **Proceso:** Llama a un LLM: *"Basado en el contexto [resumen] y las siguientes predicciones para cada continuación posible: [lista de predicciones], ¿cuál es la decisión táctica óptima? (ej. 'Execute Breakout Trade', 'Execute Bounce Trade', 'Hold/No Action'). Considera el riesgo/recompensa implícito. Justifica."*
        *   **Output:** `TacticalDecision` (objeto Pydantic con la acción decidida y justificación).
    *   **`Performance Logger Agent`:**
        *   **Input:** `TacticalDecision`, resultado real de la operación (si se ejecutó).
        *   **Proceso:** Formatea y almacena los detalles de la decisión, la justificación del LLM, los features usados, el modelo usado, la predicción y el resultado final en una base de datos estructurada (MySQL o similar) y/o como un documento en el RAG para aprendizaje futuro.
5.  **Feature Engineering Module:**
    *   **Función:** Scripts Python responsables de calcular y extraer *tanto los features básicos del STC como los features adicionales sugeridos por el `Feature Suggester Agent`*. Debe ser capaz de tomar una lista de nombres de features y devolver sus valores para un punto de datos específico.
    *   **Tecnología:** Pandas, TA-Lib, otras librerías de análisis técnico.
6.  **Predictive Model Hub:**
    *   **Función:** Un módulo que puede:
        *   Recibir una lista de features, datos históricos, un tipo de modelo (`SelectedModelType`) e hiperparámetros (inicialmente por defecto, luego optimizados).
        *   Entrenar el modelo especificado.
        *   Almacenar el modelo entrenado (ej. con MLflow, o simple serialización).
        *   Cargar un modelo entrenado y realizar una predicción sobre nuevos datos.
    *   **Tecnología:** Scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, PyTorch.
7.  **Execution Engine:**
    *   **Función:** Recibe la `TacticalDecision` final y, si implica una operación, interactúa con la API del broker/exchange o un simulador para ejecutar la orden.
    *   **Tecnología:** API del broker, CCXT, simulador propio.

**Flujo de Trabajo Detallado (Ejecución del CIPTC Orchestrator):**

1.  Recibe señal STC (Puntuación > umbral).
2.  **`Context Analyzer`:** Obtiene contexto del RAG.
3.  **`Continuation Proposer`:** LLM propone "Breakout", "Bounce", "Wait".
4.  **Para "Breakout":**
    *   **`Feature Suggester`:** LLM sugiere features F_B1, F_B2 (ej. Volatilidad reciente, fuerza de la tendencia STC).
    *   **`Model Selector`:** LLM sugiere "Gradient Boosting".
    *   **`Feature Engineering Module`:** Calcula F_B1, F_B2 y features STC.
    *   **`Predictive Model Hub`:** Entrena/Carga modelo GB_Breakout, predice Prob_Breakout.
5.  **Para "Bounce":**
    *   **`Feature Suggester`:** LLM sugiere F_R1, F_R2 (ej. Sobreventa RSI, distancia a la zona de acumulación STC).
    *   **`Model Selector`:** LLM sugiere "Logistic Regression" (quizás por ser más simple para un rebote).
    *   **`Feature Engineering Module`:** Calcula F_R1, F_R2 y features STC.
    *   **`Predictive Model Hub`:** Entrena/Carga modelo LR_Bounce, predice Prob_Bounce.
6.  **`Tactical Decision Agent`:** LLM recibe contexto, Prob_Breakout, Prob_Bounce. Decide "Execute Bounce Trade" (Pydantic).
7.  **`Execution Engine`:** Ejecuta la orden de compra para el rebote.
8.  **`Performance Logger`:** Registra todo el proceso y, más tarde, el resultado de la operación.

**Ciclo de Auto-Mejora (Fase Futura):**

*   El `Performance Logger` acumula datos sobre qué decisiones, bajo qué contextos, con qué features/modelos, llevaron a qué resultados.
*   Esta información se ingesta en el **Aia4you-Core RAG**.
*   En futuras ejecuciones, los prompts para los agentes (`Feature Suggester`, `Model Selector`, `Tactical Decision`) incluirán este *feedback histórico* del RAG.
    *   *Ejemplo Prompt para `Feature Suggester`:* "...considerando que en el pasado, el feature [X] tuvo una alta correlación con predicciones exitosas de rebote en contextos [Y]..."
*   Se puede añadir un **`Meta-Optimizer Agent`** que periódicamente analice los logs de rendimiento y sugiera cambios en los prompts de los otros agentes, en los hiperparámetros por defecto de los modelos, o incluso proponga probar nuevos tipos de continuaciones o modelos.

**Instrucciones Claras para Ti (Mi Asistente Humano):**

1.  **Configuración Inicial de la Infraestructura:**
    *   Configura la base de datos vectorial (Supabase/Pinecone) y obtén las credenciales.
    *   Configura el acceso a las APIs de LLM (Gemini, Claude) y obtén las claves.
    *   Configura el entorno Python con las librerías necesarias (Pydantic, LangChain (si la usamos), Scikit-learn, XGBoost, Pandas, TA-Lib, conectores de DB, etc.).
    *   Configura el repositorio Git para el proyecto CIPTC.
2.  **Ingesta Inicial de Datos en el RAG:**
    *   Reúne los documentos clave: Documentación STC (el script que me pasaste), README de Archon, documentación básica sobre Gradient Boosting, LSTM, Regresión Logística, y conceptos de trading (ruptura, rebote, falsa ruptura).
    *   Utiliza (o desarrolla bajo mi guía) los scripts iniciales del `IngestionService` para cargar estos documentos en la base de datos vectorial. Verifica que la ingesta sea correcta.
3.  **Provisión de Datos Históricos:**
    *   Asegúrate de que tengamos acceso a datos históricos OHLCV+Volumen fiables (formato CSV o base de datos) para los activos y timeframes que usaremos para entrenar los modelos predictivos. Prepara funciones para cargar y pre-procesar estos datos.
4.  **Interfaz del Execution Engine:**
    *   Proporciona una interfaz simple (puede ser una función Python) que el `CIPTC Orchestrator` pueda llamar para "simular" la ejecución de una orden (comprar, vender, mantener). Inicialmente, solo necesita registrar la acción. Más adelante, la conectarás a una API de broker real o un simulador más avanzado.
5.  **Feedback y Validación:**
    *   Revisa las salidas estructuradas (Pydantic) que generan los agentes LLM en las primeras ejecuciones. ¿Son coherentes? ¿Tienen sentido desde la perspectiva del trading? Tu intuición humana es vital aquí.
    *   Valida las decisiones tácticas iniciales antes de que se ejecuten automáticamente.

**Próximos Pasos para Mí (Como Arquitecto/Desarrollador Senior):**

1.  Diseñar los modelos Pydantic detallados para todas las interacciones entre agentes y con los LLMs.
2.  Escribir los prompts iniciales para cada agente (`Context Analyzer`, `Continuation Proposer`, `Feature Suggester`, `Model Selector`, `Tactical Decision`).
3.  Desarrollar la lógica central del `CIPTC Orchestrator` (probablemente usando LangGraph).
4.  Desarrollar la lógica del `Feature Engineering Module` para calcular los features básicos y prepararlo para los sugeridos.
5.  Desarrollar la interfaz del `Predictive Model Hub` para entrenamiento y predicción.
6.  Desarrollar la lógica del `Performance Logger`.

Este enfoque pone al LLM en el centro del diseño de la estrategia post-señal, utiliza el RAG para el contexto, Pydantic para la estructura, y establece un camino claro hacia la automatización y el autoaprendizaje, tal como lo solicitaste. ¡Empecemos a trabajar!