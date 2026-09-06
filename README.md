# Informe para proyecto de Predicción del GRD

Este repositorio pretende servir como repo central para el trabajo en equipo que realizaremos para la entrega del informe número 2 del ramo.

1. [Descripción del Informe 2](Proyecto2-Predicción-del-GRD.md)
2. [Formato de Presentación](MSI608-Presentacion.pptx)
3. [Informe en Markdown](informe2-paper.md)
4. [Proyecto en Python](dataset/analisis_multiclase.ipynb)
5. [Paper en Overleaf](https://es.overleaf.com/read/ykwcyxjvqfdz#1e5217)

## Explicación de Contenido
GRD (Grupos Relacionados por Diagnóstico) = sistema que clasifica cada hospitalización en uno de ~1095 grupos. Es la versión chilena de APR-DRG de 3M, que usa FONASA para pagar a los hospitales: cada egreso entra a un grupo clínica y econónomicamente similar, y el hospital cobra el "Precio FONASA" asociado. El GRD es prácticamente el resumen de toda la hospitalización.

Los archivos hacen esto:
- `dataset_elpino.csv` (14 561 filas = egresos): cada fila es una hospitalización del Hospital El Pino. Tiene hasta 35 diagnósticos (DIag01 = principal, resto = comorbilidades) en códigos CIE-10, hasta 30 procedimientos en códigos CIE-9-CM, Edad, Sexo, y la columna GRD = tu variable objetivo. Tiene 526 GRDs distintos.
- `CIE-10.xlsx` (39 873 códigos): catálogo para decodificar los diagnósticos. CIE-9.xlsx (4 646): catálogo de procedimientos.
- `IR-GRD V3.1 CON PRECIOS FONASA 2016.xlsx`: la tabla maestra de los GRDs (nombre, Peso, CDM, tipo, precio). Es la "respuesta" de la clasificación.
- Tablas maestras bases `GRD.xlsx`: tablas auxiliares (hospitales, pabellones, e severidad 0–3 y mortalidad 0–3).

**Cómo se lee un código GRD (ej. 146101 = "PH CESÁREA")**: las primeras cifras son la CDM (Categoría Diagnóstica Mayor), 14 = Embarazo, Parto y Puerperio; el medio identifica el grupo base (61 = cesárea); y el último dígito es el nivel de severidad/comorbilidad: 
- 1 sin CC, 
- 2 W/CC (con comorbilidades), 
- 3 W/MCC (comorbilidades mayores). 

Verás que 146101 / 146102 / 146103 son la misma cesárea con severidad creciente (y precio creciente: peso 0.51 → 0.82).

Es decir, el notebook convierte el problema en: dado los diagnósticos y procedimientos de un egreso (tokenizados como secuencia de 65), predecir el nivel de severidad de la cesárea (1/2/3). 

## Equipo
- Eric Silva [@eRICasl](https://github.com/eRICasl)
- Brian Guzman [@bguzmanm](https://github.com/bguzmanm)
- Miguel González [@obsesiva-syntaxis](https://github.com/obsesiva-syntaxis)
- Edson Quevedo [@braulio20aa](https://github.com/braulio20aa)
- Jaime Rivera [@jrlatin2](https://github.com/jrlatin2)

**Course:** MSI608 — Special Topics in Data Science, Master's in Computer Engineering, Universidad Andrés Bello (UNAB) **Delivery date:** Sept 12, 2026