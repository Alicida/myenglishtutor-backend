from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
import os
import logging
import base64
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import InputOutputTextPair
import logging
import asyncio
import pprint

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# ----> CONFIGURACIÓN DE CORS <----
origins = [
    "http://localhost:3000"  # Origen del frontend en localhost
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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# ----> FIN DE LA CONFIGURACIÓN DE LOGGING <----

# ----> CONFIGURACIÓN DE VERTEX AI <----
vertexai.init(project="hotline-434020", location="us-central1") # Reemplaza con tu proyecto y ubicación
# ----> FIN DE LA CONFIGURACIÓN DE VERTEX AI <----

@app.post("/transcribe/")
async def transcribe_audio(request: Request, audio_file: UploadFile = File(...)):
    print("Recibiendo audio...")
    logger.info(f"Solicitud recibida desde: {request.client.host}")
    logger.info(f"Origen de la solicitud: {request.headers.get('Origin')}")

    # Convierte el audio a texto usando Google Cloud Speech-to-Text
    client = speech.SpeechClient()
    print(client)
    content = await audio_file.read()  # Lee el contenido del archivo de audio

    ruta_archivo = os.path.join(os.getcwd(), "audio_recibido.wav")
    with open(ruta_archivo, "wb") as f:
        f.write(content)
        
    print("Tipo de contenido:", type(content))  # Imprime el tipo de dato de 'content'

    # Crea el objeto RecognitionAudio con el contenido del audio
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        # sample_rate_hertz=48000,  # Ajusta la frecuencia de muestreo si es necesario
        language_code="en-US",
    )

   # Crea una tarea asíncrona para la solicitud a la API de Speech-to-Text
    async def transcribe_task():
        try:
            response = client.recognize(config=config, audio=audio, timeout=60)
            # pprint.pprint(response)
            return response
        except Exception as e:
            print("Error al llamar a la API de Speech-to-Text:", e)
            return None

    # Ejecuta la tarea asíncrona
    task = asyncio.create_task(transcribe_task())

    # Espera a que la tarea se complete
    response = await task
   
    if response is not None:
        for result in response.results:
            # El atributo alternatives contiene una lista de posibles transcripciones
            # Usualmente la primera alternativa es la más probable
            transcript += result.alternatives[0].transcript

    print("Transcripción:", transcript)

    # ---->  AQUÍ VA LA LÓGICA PARA GENERAR LA RESPUESTA CON PALM 2 <----
    model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    response = model.predict(transcript, **parameters)
    response_text = response.text
    print("Respuesta de PaLM 2:", response_text)
    # ----> FIN DE LA LÓGICA DE PALM 2 <----

    # Convierte la respuesta de texto a audio usando Google Cloud Text-to-Speech
    synthesis_input = texttospeech.SynthesisInput(text=response_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Guarda el audio generado en un archivo temporal
    with open("response.wav", "wb") as out:
        out.write(response.audio_content)

    print("Enviando respuesta al frontend...")
    return {"audio_path": "response.wav"}