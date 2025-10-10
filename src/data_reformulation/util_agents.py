import os
import time
import codecs
from openai import OpenAI
import logging
import datetime
from tqdm import tqdm, trange

import json

json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)


def sleep(n=0, infor='sleeping!') :
    if len(infor) :
        print(infor)
    try :
        for _ in trange(n) :
            time.sleep(1.)
    except :
        for _ in trange(3) :
            time.sleep(1.)

class LLM_Agent() :
    def __init__(self, system_prompt="You are a helpful assistant.", model='deepseek-chat', max_len=8192, api_key=None, base_url=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt
        self.model = model
        self.max_len = max_len

    def i_query_withlog(self, engine, messages, temperature=0) :
        time_now = str(datetime.datetime.now())
        # response = openai.ChatCompletion.create(engine=engine, messages=messages, temperature=temperature)
        response = self.client.chat.completions.create(model=engine, messages=messages, temperature=temperature, stream=False)
        response = response.to_dict()
        
        return response

    def query(self, inputs):
        try :
            response = self.i_query_withlog(engine=self.model, messages=[{'content':inputs, 'role':'user'}])
        except :
            sleep(3)
            response = self.i_query_withlog(engine=self.model, messages=[{'content':inputs, 'role':'user'}])

        message = response["choices"][0]['message']

        return message['content']

    
if __name__ == '__main__':
    agent = LLM_Agent()
    content = agent.query('你是什么模型？我可以如何使用你？')
    print(content)






