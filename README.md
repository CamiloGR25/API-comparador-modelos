# Proyecto Chat IA - Comparación RNN vs LSTM vs Transformer

Proyecto académico para comparar tres arquitecturas de modelado de lenguaje en un contexto profesional realista usando el mismo dataset base.

## Objetivo

Entrenar y comparar tres modelos en PyTorch:

- RNN (`nn.RNN`)
- LSTM (`nn.LSTM`)
- Transformer (`nn.TransformerEncoder`)

Todos se entrenan sobre el mismo dataset profesional en formato `input/target`, transformado internamente a secuencias tipo:

```text
usuario: [entrada profesional] asistente: [respuesta esperada]
```

Esto permite luego exponer los tres en una API de chat y comparar cómo cambia la respuesta según la arquitectura.

## Dataset

Coloca los archivos del dataset en `data/raw/`:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

El proyecto ya incluye una copia inicial del dataset profesional.

Formato JSONL esperado:

```json
{
  "category": "soporte_tecnico",
  "input": "El usuario no puede acceder al correo corporativo...",
  "target": "No puede acceder al correo corporativo."
}
```

## Entrenamiento

Instala dependencias:

```bash
pip install -r requirements.txt
```

Entrena los tres modelos:

```bash
python training/train_rnn.py
python training/train_lstm.py
python training/train_transformer.py
```

Los pesos y vocabularios quedarán en `data/processed/`.

## API

Levanta la API:

```bash
uvicorn api.main:app --reload
```

Prueba un chat:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lstm",
    "message": "natalia llamó a sergio para confirmar la reunión. él solicitó moverla para el jueves. ¿quién solicitó mover la reunión?",
    "session_id": "demo1"
  }'
```

## Estructura

```text
proyecto-chat-ia/
├── api/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── training/
├── utils/
└── requirements.txt
```

## Nota metodológica

La comparación es académicamente útil porque:

- el dataset es el mismo para los tres,
- el pipeline de entrenamiento es el mismo,
- el formato de interacción es el mismo,
- y la diferencia principal proviene de la arquitectura.

## Siguiente mejora sugerida

- agregar evaluación automática en `val/test`,
- registrar métricas por categoría,
- construir frontend con selector de modelo,
- y visualizar diferencias de contexto por arquitectura.

## Evaluación comparativa

Después de entrenar los tres modelos, puedes ejecutar una evaluación común sobre `data/raw/test.jsonl`:

```bash
python evaluation/evaluate_models.py
```

Esto genera en `evaluation/results/`:

- `rnn_summary.json`, `lstm_summary.json`, `transformer_summary.json`
- `rnn_predictions.json/csv`, `lstm_predictions.json/csv`, `transformer_predictions.json/csv`
- `comparison_report.md`

### Métricas incluidas

- **Exact Match**: proporción de respuestas idénticas al objetivo.
- **Token F1**: solapamiento entre palabras de la predicción y del objetivo.
- **Métricas por categoría**: soporte técnico, servicio al cliente, sentimiento, etc.

### Ver ejemplos donde los modelos divergen

```bash
python evaluation/show_examples.py
```

Este script imprime algunos casos donde RNN, LSTM y Transformer producen respuestas distintas, para facilitar el análisis académico.
