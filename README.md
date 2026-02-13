# Worthia
Herramienta conversacional para inversionistas que facilita consultar y entender información relevante de compañías listadas. Se enfoca en análisis fundamental (modelo de negocio, métricas financieras, ROIC, márgenes, FCF, moats, riesgos) para apoyar la toma de decisiones informadas y fomentar pensamiento independiente.

## ¿Qué resuelve?
El front-end de `main_07.py` levanta la aplicación Streamlit "WorthIA", que conecta un prompt fuerte con la API de OpenAI (modelo `gpt-5.1` + `whisper-1` para transcripción + `gpt-4o-mini-tts` para texto a voz) y expone una experiencia conversational cargada de memoria, audio entrante y llamadas a herramientas externas (`tooling.handle_tool_calls`). La app mantiene el historial de chat en `st.session_state`, habilita entrada de voz, ejecuta las funciones necesarias para enriquecer la respuesta y luego transmite la respuesta final en texto y audio mientras guarda el último mensaje no streaming.

## Flujo de la solución
1. El usuario escribe o graba un mensaje en la barra lateral; si usa audio, se transcribe con `whisper-1`.
2. La conversación se agrega a la historia (`st.session_state.messages`) y se construye un prompt compuesto por `stronger_prompt` y el historial.
3. Se invoca la API de OpenAI en dos fases: primero para procesar llamadas a herramientas (la respuesta intermedia se guarda sin streaming) y después para mostrar la respuesta final en modo streaming con `stream_assistant_answer`.
4. Una vez tiene la respuesta textual, se genera audio con `gpt-4o-mini-tts`, se adjunta al mensaje y se reproduce dentro del chat.
5. Si aparecen `tool_calls`, se ejecutan con `tooling.handle_tool_calls` y se inserta cada resultado en la conversación para mantener coherencia.

## Tecnología clave
- Python 3.13+ (`pyproject.toml` obliga a esa versión).
- Streamlit para la UI (chat, inputs de audio y mensajes).
- OpenAI (chat completions, transcripciones y generación de voz).
- `python-dotenv` para inyectar `OPENAI_API_KEY` desde un archivo `.env`.

## Instalación

### Prerrequisitos generales
1. Git, Python 3.13 (o superior) y `pip`.
2. Crear un directorio de proyecto y clonar este repositorio.
3. Disponer de una clave válida de OpenAI y exportarla como `OPENAI_API_KEY`.

### macOS
```bash
bash -lc "brew update && brew install python@3.13"
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### Linux
```bash
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-distutils
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Si prefieres instalar las dependencias de forma explícita sin usar `pip install -e .`, ejecuta `pip install openai python-dotenv streamlit`.

## Configuración
1. Copia `.env.example` (si existe) o crea un `.env` nuevo.
2. Define la clave de OpenAI:
```dotenv
OPENAI_API_KEY=tu_clave_aqui
```
3. (Opcional) `OPENAI_ORGANIZATION` si trabajas con múltiples organizaciones.

## Ejecución
```bash
source .venv/bin/activate
uv run streamlit run main_07.py
```
La aplicación abrirá un servidor local en `http://localhost:8501` y podrás interactuar vía chat/voz con FinguIA.

## Qué revisar de la base
- `main_07.py`: entrypoint de la app Streamlit y la lógica de conversación.
- `prompts.py`: prompt base (`stronger_prompt`) que condiciona la conversación.
- `tooling.py` y `utils.py`: definiciones de las herramientas auxiliares.
- `pyproject.toml`: dependencias oficiales.
