import time
import os
import random
from openai import OpenAI

class Eval:
    def __init__(self, llm_api_key, base_url=None, model=None, temperature=0.3):
        self.llm_api_key = llm_api_key
        self.base_url = base_url or "http://localhost:8000/v1"
        self.model = model or "qwen3-4b"
        self.temperature = temperature
        # 使用相对路径，从当前文件向上两级，然后进入prompts目录
        self.prompt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
        
    def _call_llm_api(self, content: str, max_retries: int = 10) -> str:
        client = OpenAI(
            api_key=self.llm_api_key,
            base_url=self.base_url,
        )
        
        for attempt in range(max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.model, 
                    messages=[{"role": "user", "content": content}], 
                    temperature=self.temperature,
                    max_tokens=1  # 严格限制：只生成1个token，确保输出0或1
                )
                return completion.choices[0].message.content.strip("```").strip("json")
            except Exception as e:
                # 检查是否是上下文长度错误
                error_msg = str(e)
                if "maximum context length" in error_msg or "tokens" in error_msg:
                    print(f"\n🚨 上下文长度超限错误详情:")
                    print(f"错误信息: {error_msg}")
                    print(f"📏 Prompt长度: {len(content)} 字符")
                    print(f"📝 完整Prompt内容:")
                    print("=" * 80)
                    print(content)
                    print("=" * 80)
                                               
                    print(f"💡 建议: 考虑截断过长的内容或增加API模型的上下文长度")
                    print()
                
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"🔄 API重试 {attempt + 1}/{max_retries}, 等待{wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def _load_prompt(self, prompt_file: str) -> str:
        with open(os.path.join(self.prompt_dir, prompt_file), 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def eval_answer(self, question, gold_answer, predicted_answer):
        prompt = self._load_prompt("eval_answer_prompt.txt").format(
            question=question, gold_answer=gold_answer, predicted_answer=predicted_answer
        )
        return self._call_llm_api(prompt)
    
    def eval_query(self, question, gold_query, predicted_query):
        prompt = self._load_prompt("eval_query_prompt.txt").format(
            question=question, gold_query=gold_query, predicted_query=predicted_query
        )
        return self._call_llm_api(prompt)