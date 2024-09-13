from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import language_v1
from google.cloud import translate_v2 as translate
import os
import logging
import vertexai
from vertexai.language_models import TextGenerationModel

app = FastAPI()

# ----> CONFIGURACIÓN DE CORS <----
origins = [
    "http://localhost:3000",  # Origen del frontend en localhost
    "https://myenglishtutor-271678354785.us-central1.run.app:3000"  # Origen del frontend en Cloud Run
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----> FIN DE LA CONFIGURACIÓN DE CORS <----

# ----> CONFIGURACIÓN DE LOGGING <----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ----> FIN DE LA CONFIGURACIÓN DE LOGGING <----

# Monta el directorio "static" para archivos estáticos
app.mount("/static", StaticFiles(directory="build"), name="static")

# ----> CONFIGURACIÓN DE VERTEX AI <----
vertexai.init(project="hotline-434020", location="us-central1") # Reemplaza con tu proyecto y ubicación
# ----> FIN DE LA CONFIGURACIÓN DE VERTEX AI <----

# ----> FUNCIÓN PARA TRANSCRIBIR AUDIO <----
def transcribe_speech(content):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="es-MX",
        model="default",
        audio_channel_count=1,
        enable_word_time_offsets=True,
        alternative_language_codes=["en-US"],
    )

    # Utiliza long_running_recognize para audio de larga duración
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Esperando a que la operación se complete...")
    response = operation.result(timeout=90)  # Espera hasta 90 segundos

    transcript = ""
    for result in response.results:
        transcript += " " + result.alternatives[0].transcript
    
    transcript = transcript.strip()

    print("Transcripción:", transcript)
    return transcript
# ----> FIN DE LA FUNCIÓN PARA TRANSCRIBIR AUDIO <----

# ----> FUNCIÓN PARA ANALIZAR LA GRAMÁTICA <----
def analyze_grammar(text):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    syntax_response = client.analyze_syntax(request={'document': document})

    errors = []
    for token in syntax_response.tokens:
        # Ejemplo de detección de error: verbo en tercera persona del singular sin "s" al final
        if token.part_of_speech.tag == 'VERB' and token.dependency_edge.label == 'ROOT' and token.text.endswith("ar") and not token.text.endswith("as"):
            errors.append(f"Error gramatical: El verbo '{token.text}' debería terminar en 's' en tercera persona del singular.")

    return errors
# ----> FIN DE LA FUNCIÓN PARA ANALIZAR LA GRAMÁTICA <----

# ----> FUNCIÓN PARA DETECTAR IDIOMA <----
def detect_language(text):
    translate_client = translate.Client()
    result = translate_client.detect_language(text)
    return result['language']
# ----> FIN DE LA FUNCIÓN PARA DETECTAR IDIOMA <----

@app.post("/transcribe/")
async def transcribe_audio(request: Request, audio_file: UploadFile = File(...)):
    print("Recibiendo audio...")
    logger.info(f"Solicitud recibida desde: {request.client.host}")
    logger.info(f"Origen de la solicitud: {request.headers.get('Origin')}")

    content = await audio_file.read()  # Lee el contenido del archivo de audio

    # Guarda el audio recibido en un archivo temporal
    ruta_archivo = os.path.join(os.getcwd(), "audio_recibido.wav")
    with open(ruta_archivo, "wb") as f:
        f.write(content)

    print("Tamaño del archivo guardado:", os.path.getsize(ruta_archivo))

    # Llama a la función transcribe_speech para obtener la transcripcion
    transcript = transcribe_speech(content)

    # ----> ANÁLISIS DE GRAMÁTICA <----
    errors = analyze_grammar(transcript)
    if errors:
        for error in errors:
            print(error)
    # ----> FIN DEL ANÁLISIS DE GRAMÁTICA <----

    # ---->  AQUÍ VA LA LÓGICA PARA GENERAR LA RESPUESTA CON PALM 2 <----
    model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    prompt = f"""You are an English teacher. The user said: '{transcript}'. 
    If there are any grammatical errors, correct them and provide an explanation in natural language, written in English. 
    Continue the conversation or ask me about the errors the user made. 
    If what the user said doesn't make sense, try to deduce what they meant. 
    Please format your response as a single paragraph without using any bullet points or asterisks."""
    response = model.predict(prompt, **parameters)
    response_text = response.text
    print("Respuesta de PaLM 2:", response_text)
    # ----> FIN DE LA LÓGICA DE PALM 2 <----

    # ----> DETECTA EL IDIOMA DE LA RESPUESTA <----
    language_code = detect_language(response_text)
    print(f"Idioma detectado: {language_code}")
    # ----> FIN DE LA DETECCIÓN DE IDIOMA <----

    # Convierte la respuesta de texto a audio usando Google Cloud Text-to-Speech
    synthesis_input = texttospeech.SynthesisInput(text=response_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,  # Usa el idioma detectado
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Guarda el audio generado en el directorio "static"
    with open("build/response.wav", "wb") as out:
        out.write(response.audio_content)

    print("Enviando respuesta al frontend...")
    return {"audio_path": "/static/response.wav"}