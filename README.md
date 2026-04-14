# Tabulens

**Tabulens** es una librería de Python diseñada para **perfilar, validar, limpiar, filtrar y optimizar DataFrames**, con el objetivo de mejorar la calidad de los datos de manera estructurada y reproducible antes de su análisis.

---

## Instalación

```bash
pip install tabulens
```

---

## Descripción

El análisis de datos en la práctica suele implicar tareas repetitivas como:

* detección de valores nulos
* validación de reglas de negocio
* limpieza de datos inconsistentes
* eliminación de duplicados
* optimización de memoria

Estas tareas normalmente se realizan de forma manual, lo que introduce errores y falta de consistencia.

**Tabulens** propone un flujo unificado que permite:

* centralizar reglas de validación
* aplicar estrategias de limpieza de forma declarativa
* obtener reportes claros sobre el estado de los datos
* separar datos válidos e inválidos de forma programática

---

## Uso básico

```python
import pandas as pd
from tabulens import Tabulens

df = pd.read_csv("data.csv")

t = Tabulens(df)

# Profiling
profile = t.profile()
print(profile.render_text())

# Validation
validation = (
    t.validate()
    .not_null("edad")
    .in_range("edad", min_value=18, max_value=99)
    .run()
)

print(validation.render_text())

# Cleaning
cleaned = t.clean(
    null_strategy={"edad": "median"},
    duplicate_strategy="drop"
)

# Filtering
valid_df = t.keep_valid_rows(validation)

# Optimization
optimized = t.optimize()

# Insights
insights = t.insights()
print(insights.render_text())
```

---

## Módulos

### Profiling

Analiza la estructura del DataFrame y genera información descriptiva como:

* tipos de datos
* valores nulos
* cardinalidad
* posibles recomendaciones estructurales

### Validation

Permite definir reglas sobre columnas:

* `not_null`
* `unique`
* `in_range`
* `allowed_values`
* `regex`

Produce un reporte que incluye:

* cumplimiento por regla
* índices de filas que fallan
* identificación de casos cercanos a incumplimiento

### Cleaning

Permite aplicar estrategias de limpieza de forma selectiva por columna.

Estrategias disponibles:

* `mean`, `median`, `mode`
* `ffill`, `bfill`
* `fill_value`
* `drop_rows`

También permite eliminación de duplicados.

### Filtering

A partir de un reporte de validación, permite:

* conservar únicamente filas válidas
* conservar filas inválidas
* separar ambos conjuntos

### Optimization

Optimiza el uso de memoria mediante:

* conversión de tipos numéricos
* detección de columnas categóricas

### Insights

Genera observaciones simples sobre el comportamiento de los datos.

---

## Notebook tutorial

El uso completo de la librería se muestra en el siguiente notebook:

**[Open in Colab](https://colab.research.google.com/github/DavidVarH/tabulens/blob/main/notebooks/tabulens_tutorial.ipynb)**

El notebook:

* instala la librería desde PyPI
* muestra un flujo completo de uso
* es reproducible en Google Colab

---

## Tests

Para ejecutar los tests:

```bash
pytest
```

---

## Estructura del proyecto

```
tabulens/
│
├── profiling.py
├── validation.py
├── cleaning.py
├── filtering.py
├── optimization.py
├── insights.py
├── rules.py
├── utils.py
│
tests/
```

---

## Objetivo del proyecto

El objetivo de Tabulens es ofrecer una herramienta práctica para mejorar la calidad de los datos antes de su análisis, reduciendo errores y estandarizando procesos comunes en ciencia de datos.

---

## Licencia

MIT License
