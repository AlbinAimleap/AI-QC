import boto3
import json
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
import argparse

class BedrockTextGenerator:
    def __init__(self):
        load_dotenv(find_dotenv(".env"))
        self.bedrock = self._connect_bedrock()
        
    def _connect_bedrock(self):
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def _construct_prompt(self, prompt, context=None):
        if context:
            with open(context, 'r') as file:
                context_content = file.read()
            full_prompt = f"Context:\n{context_content}\n\nfilename:{context}\n\nPrompt: {prompt}"
        else:
            full_prompt = prompt
        return full_prompt
    
    def generate_text(self, prompt, context_file=None, max_tokens=1000):
        try:
            constructed_prompt = self._construct_prompt(prompt, context_file)
            response = self.bedrock.invoke_model(
                modelId='us.anthropic.claude-3-5-sonnet-20240620-v1:0',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {"role": "user", "content": constructed_prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0
                })
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return None
        
        
    def load_prompt(self, prompt_file, checklist_file):
        prompt_file_path = Path(prompt_file)
        checklist = Path(checklist_file)
        with open(prompt_file_path, 'r') as file:
            prompt = file.read()
        
        with open(checklist, "r") as file:
            checklist = file.read()
            return prompt.replace("{{checklist}}", checklist)

class QualityChecker:
    def __init__(self):
        self.generator = BedrockTextGenerator()

    def argument_parser(self):
        parser = argparse.ArgumentParser(description="QC code using Bedrock.")
        parser.add_argument("--context", nargs='+', type=str, help="The context files to use.")
        parser.add_argument("--prompt", type=str, help="The prompt to generate text for.", default="prompt.txt")
        parser.add_argument("--checklist", type=str, help="The context file to use.", default="checklist.txt")
        return parser.parse_args()

    def run(self):
        args = self.argument_parser()
        prompt = self.generator.load_prompt(args.prompt, args.checklist)
        
        for context_file in args.context:
            response = self.generator.generate_text(prompt, context_file)
            if response:
                print(response)
            print("-" * 50)

if __name__ == "__main__":
    qc = QualityChecker()
    qc.run()