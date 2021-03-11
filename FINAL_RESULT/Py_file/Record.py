def Record_Stress():
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.preprocessing.image import ImageDataGenerator
    import warnings
    warnings.filterwarnings(action='ignore')

    audio_data = 'output.wav'
    y , sr = librosa.load(audio_data)
    librosa.feature.melspectrogram(y=y, sr=sr)
    
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
    plt.axis('off')
    plt.savefig('audio_test/0/output.png')
#     plt.clf()
    
    test_datagen = ImageDataGenerator(
        rescale=1./255) # image will be flipper horiztest_datagen = ImageDataGenerator(rescale=1./255)  
    test_image = test_datagen.flow_from_directory(
        'audio_test/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)
    
    return test_image