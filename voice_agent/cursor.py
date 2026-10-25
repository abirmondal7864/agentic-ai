import asyncio
import io
import json
import os
import subprocess
import time
import wave
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import requests
import speech_recognition as sr

# Suppress ALSA / Pygame noisy startup banners on Linux
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def play_audio(pcm_bytes: bytes):
    """Wraps raw 24kHz PCM bytes into a WAV container and plays them."""
    import pygame

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(24000)  # 24 kHz
        wav_file.writeframes(pcm_bytes)

    wav_io.seek(0)

    pygame.mixer.init()
    sound = pygame.mixer.Sound(wav_io)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.delay(50)
    pygame.mixer.quit()


async def tts(speech: str):
    """Synthesizes speech using gemini-3.1-flash-tts-preview and plays it."""
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=speech,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"
                    )
                )
            ),
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            await asyncio.to_thread(play_audio, part.inline_data.data)


def run_command(cmd: str) -> str:
    """Executes system command and captures stdout/stderr string."""
    return subprocess.getoutput(cmd)


def get_weather(city: str) -> str:
    """Fetches simple weather info from wttr.in."""
    try:
        url = f"https://wttr.in/{city.lower()}?format=%C+%t"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}"
        return "Weather service returned an error"
    except Exception as e:
        return f"Failed to fetch weather: {e}"


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
}

SYSTEM_PROMPT = """
You are an expert AI assistant in resolving user queries using chain of thought.
You work on START, PLAN, TOOL, and OUTPUT steps.
You need to first PLAN what needs to be done. The PLAN can be in multiple steps.
Once you think enough PLAN has been done, finally give an OUTPUT.
You can call a tool if required from the list of available tools.
For every tool call, wait for the OBSERVE step which is the output from the called tool.

Rules:
- Strictly follow the given JSON output format.
- Output only ONE step at a time as a single valid JSON object.
- The step attribute must be one of: "START", "PLAN", "TOOL", "OUTPUT".

Available Tools:
- get_weather(city: str): Takes city name as an input string and returns the weather info about the city.
- run_command(cmd: str): Takes a command string as an input and executes it on the system and returns the output of the command.
"""


class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step: PLAN, OUTPUT, TOOL, START")
    content: Optional[str] = Field(None, description="The textual content or reasoning for this step")
    tool: Optional[str] = Field(None, description="The name of the tool to call")
    input: Optional[str] = Field(None, description="The input parameters for the tool")


def run_agent():
    r = sr.Recognizer()

    message_history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"System Instructions:\n{SYSTEM_PROMPT}")]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text='{"step":"PLAN","content":"Understood. I will plan step-by-step and return JSON."}')]
        )
    ]

    with sr.Microphone() as source:
        print("Calibrating microphone for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=0.8)
        r.pause_threshold = 1.5

        while True:
            print("\nListening for voice input...")
            try:
                audio = r.listen(source, timeout=6, phrase_time_limit=10)
                print("Transcribing audio...")
                user_query = r.recognize_google(audio)
            except sr.WaitTimeoutError:
                continue
            except (sr.UnknownValueError, sr.RequestError) as e:
                print(f"STT Error: {e}")
                continue

            print(f"User: '{user_query}'")
            if user_query.lower().strip() in ["exit", "quit", "stop"]:
                print("Exiting agent loop.")
                break

            message_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"START: {user_query}")]
                )
            )

            # Inner reasoning / planning / tool loop
            while True:
                response = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=message_history,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=MyOutputFormat,
                        temperature=0.2,
                    ),
                )

                raw_result = response.text
                message_history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=raw_result)]
                    )
                )

                try:
                    parsed_result = MyOutputFormat.model_validate_json(raw_result)
                except Exception as err:
                    print("JSON Parse error:", err)
                    break

                if parsed_result.step == "START":
                    print("🔥", parsed_result.content)
                elif parsed_result.step == "PLAN":
                    print("🧠", parsed_result.content)
                elif parsed_result.step == "OUTPUT":
                    print("🤖 Final Output:", parsed_result.content)
                    if parsed_result.content:
                        asyncio.run(tts(speech=parsed_result.content))
                    break
                elif parsed_result.step == "TOOL":
                    tool_to_call = parsed_result.tool
                    tool_input = parsed_result.input
                    print(f"⚒️ Calling Tool: {tool_to_call} with input: '{tool_input}'")

                    if tool_to_call in available_tools:
                        tool_response = available_tools[tool_to_call](tool_input or "")
                    else:
                        tool_response = f"Tool '{tool_to_call}' not found."

                    print(f"⚒️ Tool Result: {tool_response}")

                    message_history.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(
                                text=json.dumps({
                                    "step": "OBSERVE",
                                    "tool": tool_to_call,
                                    "input": tool_input,
                                    "output": str(tool_response)
                                })
                            )]
                        )
                    )
                    continue

                # Instruct the model to move to the next reasoning step
                message_history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="continue with next step")]
                    )
                )
                time.sleep(0.5)


if __name__ == "__main__":
    run_agent()