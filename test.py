import subprocess
import whisper
#import openai
import os
import sys
from fastapi import FastAPI, File, UploadFile
import aiofiles
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate,LLMChain
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
#openai.api_key = ""

model = whisper.load_model("base")

app = FastAPI()

def video_to_audio(video_file):
    audio_file = "input_audio.mp3"
    subprocess.call(["ffmpeg", "-y", "-i", video_file, audio_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return audio_file

def audio_to_transcript(audio_file):
    result = model.transcribe(audio_file)
    transcript = result["text"]
    return transcript

def MoM_generation(prompt):
    template = """<|prompter|>{question}<|endoftext|>
    <|assistant|>"""

    promp = PromptTemplate(template=template, input_variables=["question"])

    llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
                         model_kwargs={"max_new_tokens":100})

    llm_chain = LLMChain(
        llm=llm,
        prompt=promp
    )
    preset="Can you generate the Minute of Meeting in form of bullet points for the below transcript?\n"+prompt
    r=llm_chain.run(preset)
    print(r)

    #response = openai.Completion.create(model="text-davinci-003",
     #                                   prompt= "Can you generate the Minute of Meeting in form of bullet points for the below transcript?\n"+prompt,
      #                                  temperature=0.7,
       #                                 max_tokens=256,
          #                              top_p=1,
         #                               frequency_penalty=0,
        #                                presence_penalty=0)
    return r
        #response['choices'][0]['text']

audio_file = video_to_audio('C:\\Users\\nagar\\Downloads\\videoplayback.mp4')
print("hi")
transcript = audio_to_transcript(audio_file)
print(transcript)
final_result = MoM_generation(transcript[:1020])
print("test1")

print(final_result)

