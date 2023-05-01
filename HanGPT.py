# importing the module
import speech_recognition as sr
import openai
from io import BytesIO
import subprocess
import langid
from zhon import hanzi
import re
import cv2
import torch
# 导入translators库
import translators as ts

# 加载预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 定义一个列表，存储多个api key轮询使用，避免单个key一分钟3次的限制影响
api_keys = ["sk-xxx1","sk-xxx2"]
# 定义一个变量，记录当前使用的api key的索引
current_index = 0
openai.api_key = api_keys[current_index]
messages = []

def transcribe_audio(audio): 
    # 创建一个字节流对象，用于存储音频的wav数据
    wav_data = BytesIO(audio.get_wav_data()) 
    # 给字节流对象命名，方便传递给openai.Audio.transcribe方法 
    wav_data.name = "SpeechRecognition_audio.wav"
    # 调用openai.Audio.transcribe方法，将音频转换为文本 
    listenText = openai.Audio.transcribe("whisper-1", wav_data, api_key=api_keys[current_index],language="zh")
    # 获取转换后的文本 
    sayText = listenText["text"]
    # 返回文本 
    return sayText

# 调用chatgpt对话
def chat(prompt):
  # 声明全局变量
  global api_keys, current_index, messages
  # 将用户输入添加到对话列表中
  try:
      messages.append({"role": "user", "content": prompt})
      # 调用chatgpt3.5模型，传入对话列表
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
      )
      # 获取回答内容
      answer = response.choices[0].message["content"]
      # 将回答内容添加到对话列表中
      messages.append({"role": "assistant", "content": answer})
      # 打印回答内容
      print("ChatGPT3.5:", answer)
      return answer
  except openai.error.RateLimitError as e:
      messages.pop()
      current_index = (current_index + 1) % len(api_keys)
      return chat(prompt)

# 定义一个函数来检测文本是否具有中文语言意义
def has_chinese_meaning(text):
  # 检测文本是否包含中文句子模式
  if re.search(hanzi.sentence, text):
    return True
  lang, confidence = langid.classify(text)
  if lang == 'zh':
    return True
  # 如果都没有，则返回False
  return False

def getPicThings():
    # 打开摄像头
    capture = cv2.VideoCapture(0)
    set_all = []
    for i in range(5):
        cv2.startWindowThread()
        # 获取一帧图像
        ret, frame = capture.read()

        # 将图像传入模型进行推理
        results = model(frame)
        df = results.pandas().xyxy[0]

        # 过滤噪音
        df = df[df['name'] != 'person']
        df = df[df['name'] != 'remote']
        items = df['name'].tolist()
        if items:
            set_all.extend(items)
    return set(set_all)

# 判断是否要求开启摄像头
def isShowPic(sayText):
    trimStr = sayText.replace(" ", "").replace("?", "").replace("？", "")
    if trimStr == '这是什么' or trimStr == '這是什麼':
        return True
    else:
        return False


# 说出摄像头看到的物体名称
def sayPicThings():
    things = getPicThings()
    messages.append({"role": "user", "content": "这是什么"})
    result_with_delimiter = ''
    if things:
        result_with_delimiter = '我看到了'
        delimiter = '一个'
        for str in things:
            result_with_delimiter = result_with_delimiter + delimiter + ts.translate_text(str, to_language="zh")
    else:
        result_with_delimiter = '我什么都没看到'
    messages.append({"role": "assistant", "content": result_with_delimiter})
    print("AI:", result_with_delimiter)
    return result_with_delimiter


# 获取麦克风
r = sr.Recognizer()
mic = sr.Microphone(device_index=0)

while True:
    # 监听麦克风
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("===请说话===")
        audio = r.listen(source)
    # 转换语音流
    sayText = transcribe_audio(audio)
    if sayText:
        print("我说:",sayText)
        if has_chinese_meaning(sayText):
            returnText = ''
            if isShowPic(sayText):
                returnText = sayPicThings()
            else:
                returnText = chat(sayText)
            subprocess.run(["say", returnText])
        else:
            print("这句话没有具体的意义")
    else:
         print("没有听到任何内容")
