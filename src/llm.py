import os



from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

load_dotenv()
key = os.environ.get('GIGACHAT_API_KEY')

llm = GigaChat(credentials=key,
                model='GigaChat-2-Max',
                scope='GIGACHAT_API_CORP',
                temperature = 0,
                verify_ssl_certs = False,
                profanity_check=False,
                timeout=300)