def Speech_to_text():
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    
    audio=sr.AudioFile('output.wav')
    with audio as source:
        vocal = recognizer.record(source)
    txt = recognizer.recognize_google(audio_data= vocal, language="ko-KR")
    return txt