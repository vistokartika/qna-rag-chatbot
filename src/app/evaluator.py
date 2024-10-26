import os

from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.evaluation import EvaluatorType, load_evaluator

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

score_criteria = {
    "helpfulness": """
        The given score is between 1 and 10, where 10 is the best possible score.
        The score is calculated by comparing the generated answer with the user input. If the answer is relevant and helpful, the score will be higher.
    """,
}

helpfulness_evaluator = load_evaluator("score_string", score_criteria=score_criteria, llm=ChatOpenAI(model="gpt-4o-mini", temperature=0))