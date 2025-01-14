"""
!/usr/bin/python3.10
-*- coding: utf-8 -*-
@Time    : 2025/1/13 22:01
@Author  : wonderbell
@Email   : 969064814@qq.com
@File    : hello_agent_day1.py
@Software: PyCharm
@description: day01 agen 初见
"""
import asyncio
import configparser

from openai import OpenAI
from base_prompt import *


def read_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path, encoding='utf-8')

    app_config = {
        "api_key": config.get("deepseek", "api_key"),
        "endpoints": config.get("deepseek", "endpoints"),
    }

    return app_config


def deepseek_api(api_key, endpoints="https://api.deepseek.com"):
    client = OpenAI(api_key=api_key, base_url=endpoints)

    response = client.chat.completions.create(
        model="deepseek-chat",  # 最新 DeepSeek-V3
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    print(response.choices[0].message.content)


class SmartAssistant:
    def __init__(self, api_key, endpoints, chat_model, sys_prompt, registered_prompt, query_prompt, delete_prompt):
        self.client = OpenAI(api_key=api_key, base_url=endpoints)

        self.system_prompt = sys_prompt
        self.registered_prompt = registered_prompt
        self.query_prompt = query_prompt
        self.delete_prompt = delete_prompt
        self.chat_model = chat_model
        # Using a dictionary to store different sets of messages
        self.messages = {
            "system": [{"role": "system", "content": self.system_prompt}],
            "registered": [{"role": "system", "content": self.registered_prompt}],
            "query": [{"role": "system", "content": self.query_prompt}],
            "delete": [{"role": "system", "content": self.delete_prompt}]
        }

        # Current assignment for handling messages
        self.current_assignment = "system"

    async def get_response(self, user_input):
        self.messages[self.current_assignment].append({"role": "user", "content": user_input})
        while True:
            # 异步调用 OpenAI 客户端
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=self.messages[self.current_assignment],
                temperature=0.9,
                stream=False,
                max_tokens=2000,
            )

            ai_response = response.choices[0].message.content
            if "registered workers" in ai_response:
                self.current_assignment = "registered"
                print("意图识别:", ai_response)
                print("switch to <registered>")
                self.messages[self.current_assignment].append({"role": "user", "content": user_input})
            elif "query workers" in ai_response:
                self.current_assignment = "query"
                print("意图识别:", ai_response)
                print("switch to <query>")
                self.messages[self.current_assignment].append({"role": "user", "content": user_input})
            elif "delete workers" in ai_response:
                self.current_assignment = "delete"
                print("意图识别:", ai_response)
                print("switch to <delete>")
                self.messages[self.current_assignment].append({"role": "user", "content": user_input})
            elif "customer service" in ai_response:
                print("意图识别:", ai_response)
                print("switch to <customer service>")
                self.messages["system"] += self.messages[self.current_assignment]
                self.current_assignment = "system"
                return ai_response
            else:
                self.messages[self.current_assignment].append({"role": "assistant", "content": ai_response})
                return ai_response

    async def start_conversation(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting conversation.")
                break

            # 异步调用 get_response
            response = await self.get_response(user_input)
            print("Assistant:", response)


async def main():
    # 使用示例
    config_path = "config.ini"
    api_config = read_config(config_path)

    api_key = api_config.get("api_key", None)
    endpoints = api_config.get("endpoints", None)
    assert api_key is not None
    # deepseek_api(api_key, endpoints)
    assistant = SmartAssistant(api_key, endpoints, "deepseek-chat", sys_prompt, registered_prompt, query_prompt,
                               delete_prompt)
    await assistant.start_conversation()


if __name__ == '__main__':
    asyncio.run(main())
