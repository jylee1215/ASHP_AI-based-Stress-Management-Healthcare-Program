def sentiment_predict(new_script):
    
    from tensorflow.keras.models import load_model
    from konlpy.tag import Okt
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import numpy as np
    import pickle

    loaded_model = load_model('best_model.h5')
    okt = Okt()
    max_len = 50
    # tokenizer = Tokenizer(7285, oov_token = 'OOV') 
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

    with open('Pickle/xtrain.pickle', 'rb') as f:
        X_train = pickle.load(f)
    with open('Pickle/xtext.pickle', 'rb') as f:
        X_test = pickle.load(f)
    with open('Pickle/ytest.pickle', 'rb') as f:
        y_test = pickle.load(f)
    with open('Pickle/ytrain.pickle', 'rb') as f:
        y_train = pickle.load(f)
    with open('Pickle/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    new_script = okt.morphs(new_script, stem=True, norm = True) # 토큰화
    new_script = [word for word in new_script if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_script]) # 정수 인코딩

    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = np.array(loaded_model.predict(pad_new)) # 예측
    result = ""
    percents = ["중립", "공포", "놀람","짜증", "행복", "슬픔", "분노"]
    df = pd.DataFrame({"감정": percents, "음성_확률" : score[0]})

    return df