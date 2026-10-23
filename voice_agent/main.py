import asyncio
import io
import os
import wave
from dotenv import load_dotenv
from google import genai
from google.genai import types
import speech_recognition as sr

# Suppress ALSA / Pygame noisy startup banners
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def play_audio(pcm_bytes: bytes):
    """Wraps raw 24kHz PCM bytes into a WAV container and plays them."""
    import pygame

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(24000)  # 24 kHz
        wav_file.writeframes(pcm_bytes)

    wav_io.seek(0)

    pygame.mixer.init()
    sound = pygame.mixer.Sound(wav_io)
    sound.play()
    while pygame.mixer.get_busy():
        pygame.time.delay(50)
    pygame.mixer.quit()


def get_speech() -> str:
    """Captures microphone speech with STT."""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # Short calibration to keep turns snappy
            r.adjust_for_ambient_noise(source, duration=0.6)
            r.pause_threshold = 1.5

            print("\nListening... (Speak now, or say 'exit' to quit)")
            audio = r.listen(source, timeout=6, phrase_time_limit=10)

            print("Transcribing audio...")
            return r.recognize_google(audio)
    except sr.WaitTimeoutError:
        return ""
    except (sr.UnknownValueError, sr.RequestError):
        return ""
    except Exception as e:
        print(f"Microphone error: {e}")
        return ""


async def main():
    print("=== Continuous Voice Agent Started ===")
    print("Say 'exit', 'quit', or 'stop' to end the session.\n")

    try:
        while True:
            # 1. Listen for user voice input
            user_text = await asyncio.to_thread(get_speech)
            if not user_text:
                continue

            print(f"You said: '{user_text}'")

            # Exit condition check
            if user_text.lower().strip() in ["exit", "quit", "stop", "bye", "goodbye"]:
                print("Ending conversation session. Goodbye!")
                break

            # 2. General Reasoning: gemini-3.6-flash
            print("Thinking...")
            chat_response = await client.aio.models.generate_content(
                model="gemini-3.6-flash",
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are an expert voice agent. Respond naturally, conversationally, "
                        "and concisely in 1 to 2 sentences."
                    )
                ),
            )
            ai_text = chat_response.text
            print(f"AI response: {ai_text}")

            # 3. Speech Synthesis: gemini-3.1-flash-tts-preview
            print("Speaking...")
            tts_response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-tts-preview",
                contents=ai_text,
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

            # 4. Play the synthesized voice
            for part in tts_response.candidates[0].content.parts:
                if part.inline_data:
                    await asyncio.to_thread(play_audio, part.inline_data.data)

    except KeyboardInterrupt:
        print("\nSession stopped by user.")


if __name__ == "__main__":
    asyncio.run(main())