# 문장을 인덱스로 변환
def convert_text_to_index(sentences, vocabulary, type):
    import numpy as np
    DECODER_INPUT  = 1
    DECODER_TARGET = 2

    PAD = "<PADDING>"   # 패딩
    STA = "<START>"     # 시작
    END = "<END>"       # 끝
    OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

    max_sequences = 50
    sentences_index = [] #문장 -> 단어 -> 인덱스 -> 저장

  # 모든 문장에 대해서 반복
    for sentence in sentences:
        sentence_index = [] #각각의 문장을 구성하는 단어들에 대한 index가 저장되는 리스트 변수
    
    # 디코더 입력일 경우 맨 앞에 START 태그 추가
        if type == DECODER_INPUT:
              sentence_index.extend([vocabulary[STA]])

    # 문장의 단어들을 띄어쓰기로 분리
        for word in sentence.split(): #[12시, 땡]
            if vocabulary.get(word) is not None: #파이썬 문법이 딕셔너리는 키에대한 값이 없으면 none이 나온다.
        # 사전에 있는 단어면 해당 인덱스를 추가
              sentence_index.extend([vocabulary[word]])
        else: # 사전에 없는 단어면 OOV 인덱스를 추가
              sentence_index.extend([vocabulary[OOV]])

    # 최대 길이 검사
        if type == DECODER_TARGET:
      # 디코더 목표일 경우 맨 뒤에 END 태그 추가
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
            
    # 최대 길이에 없는 공간은 패딩 인덱스로 채움
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]

        sentences_index.append(sentence_index)

    return np.array(sentences_index)

def pos_tag(sentences):
    from konlpy.tag import Okt
    import re
    RE_FILTER = re.compile("[.,!?\"':;~()…]")

    # KoNLPy 형태소분석기 설정
    tagger = Okt()
    # 문장 품사 변수 초기화
    sentences_pos = []
    for sentence in sentences:
        # 특수기호 제거
        sentence = re.sub(RE_FILTER, "", sentence)

        sentence = " ".join(tagger.morphs(sentence))
        sentences_pos.append(sentence)

    return sentences_pos

def word_to_index_create():
    import pickle
    
    with open('word_to_index.pickle','rb') as f:
          word_to_index=pickle.load(f)
    return word_to_index

def make_predict_input(sentence):
    
    from konlpy.tag import Okt
    import re
    import pickle
    import numpy as np
    word_to_index=word_to_index_create()
    ENCODER_INPUT  = 0
    
    
    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)
    
    return input_seq

def index_to_word_create():
    import pickle
    with open('index_to_word.pickle','rb') as f:
        index_to_word=pickle.load(f)
    return index_to_word

# 인덱스를 문장으로 변환
def convert_index_to_text(indexs, vocabulary): 
    STA_INDEX = 1
    PAD_INDEX = 0
    END_INDEX = 2
    OOV_INDEX = 3
    
    sentence = ''
    
    # 모든 문장에 대해서 반복
    for index in indexs:
        if index == END_INDEX:
            # 종료 인덱스면 중지
            break;
        if vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 OOV 단어를 추가
            sentence.extend([vocabulary[OOV_INDEX]])
            
        # 빈칸 추가
        sentence += ' '

    return sentence

def load_model(model_filename, model_weights_filename):
    from keras.models import model_from_json
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

def encoder_model_create():
    encoder_model = load_model('encoder_model.json', 'encoder_model_weights.h5')
    return encoder_model
def decoder_model_create():  
    decoder_model = load_model('decoder_model.json', 'decoder_model_weights.h5')
    return decoder_model

# 텍스트 생성
def generate_text(input_seq):
    STA_INDEX = 1
    PAD_INDEX = 0
    END_INDEX = 2
    OOV_INDEX = 3
    ENCODER_INPUT  = 0
    DECODER_INPUT  = 1
    DECODER_TARGET = 2
    max_sequences = 50
    
    encoder_model=encoder_model_create()
    decoder_model=decoder_model_create()
    index_to_word=index_to_word_create()
    #array([[156, 414,  89, 405, 280, 158,   0,   0,   0,   0,   0,   0,   0,
    #      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    #      0,   0,   0,   0]])
    
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
        
    return sentence