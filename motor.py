# abhi.py
import os
from dotenv import load_dotenv
import tiktoken
from datetime import datetime
from openai import AzureOpenAI
from pathlib import Path

load_dotenv()


class ABHI:
    def __init__(self, message=None):
        self.API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.API_VERSION = "2024-02-15-preview"

        self.client = AzureOpenAI(
            api_key=self.API_KEY,
            azure_endpoint=self.RESOURCE_ENDPOINT,
            api_version=self.API_VERSION,
        )

        self.encoding = tiktoken.encoding_for_model("gpt-4-0613")
        self.messages = message if message else []

    def num_tokens_from_string(self, string: str) -> int:
        return len(self.encoding.encode(string))

    def reset_system_message(self):
        current_file_path = Path(__file__).resolve()
        current_directory = current_file_path.parent
        prompt_path = os.path.join(
            current_directory, "prompts", "main_prompts2.txt"
        ).replace("//", "/")
        with open(prompt_path, "r", encoding="utf-8") as f:
            text = f.read()
        today = datetime.now().strftime("%A, %B %d, %Y")
        self.messages = [
            {"role": "system", "content": f"{text}\n\nToday's date is {today}."}
        ]

    def ensure_message_length_within_limit(
        self, max_response_tokens=250, token_limit=50000
    ):
        def calculate_token_length(messages):
            return (
                sum(len(self.encoding.encode(m["content"])) + 3 for m in messages) + 3
            )

        while (
            calculate_token_length(self.messages) + max_response_tokens >= token_limit
        ):
            if len(self.messages) > 1:
                del self.messages[1]
            else:
                break

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        self.ensure_message_length_within_limit()
        try:
            response = self.client.chat.completions.create(
                model=self.DEPLOYMENT_NAME,
                messages=self.messages,
                max_tokens=800,
                temperature=0.7,
            )
            assistant_message = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": assistant_message})
            return assistant_message
        except Exception as e:
            return f"An error occurred: {e}"
