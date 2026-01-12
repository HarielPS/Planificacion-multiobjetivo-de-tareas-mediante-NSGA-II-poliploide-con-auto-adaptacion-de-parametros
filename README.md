# Proyecto Final — Planificación de tareas (NSGA-II poliploide)

Implementación modular de NSGA-II con **individuo multiploide (6 cromosomas)** para el problema de planificación de tareas
(makespan + consumo energético) y **auto-adaptación de parámetros** de cruza/mutación.

## Requisitos

- Python 3.10+
- numpy
- matplotlib (para gráficas)
- (Opcional) pymoo (para hipervolumen)

Instalación:

```bash
pip install -r requirements.txt
```

## Estructura

- `src/` código fuente
- `data/` escenarios `.txt`
- `outputs/` salidas por escenario(tablas, gráficas, gantt)

## Ejecución rápida

```bash
python -m src.main --scenario data/Escenario1.txt --runs 30
python -m src.main --scenario data/Escenario2.txt --runs 30
python -m src.main --scenario data/Escenario3.txt --runs 30
```

Para correr los 3 escenarios:
```bash

python -m src.main --scenario all --runs 30
```
