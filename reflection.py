import os

from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def first_answer(question: str) -> str:
    answer = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "You are a data analyst senior"},
            {"role": "user", "content": question},
        ],
    )
    return answer.choices[0].message.content or ""

def think_about_answer(question: str, answer: str) -> dict:
    thinking_prompt = f"""
    You are a technical reviser senior and mature. Analise the answer below
    
    Original Question: {question}
    
    Answer generated : {answer}
    
    return in json : 
    - "approved" (true/false)
    - "problems" (a list of concerns about the answer)
    - "suggestions"(a list of suggestions to improve the answer)
    """

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": thinking_prompt}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content) or {}

def review_answer(question: str, answer: str, review : dict) -> str:
    review_prompt = f"""
    Some questions were raised regarding your response. Rewrite it, correcting all the issues pointed out. 
    
    Original Question: {question}    
    Answer generated : {answer}
    Problems : {review['problems']}
    Suggestions : {review['suggestions']}
    """

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": review_prompt}]
    )

    return response.choices[0].message.content or ""

def reflection_agent(question: str, max_iterations: int = 3) -> str:
    response = first_answer(question)

    for interation in range(max_iterations):
        review = think_about_answer(question, response)
        if review["approved"]:
            print(f"Approved at {interation+1} interactions")
            return response

        print(f"Review at {interation+1} interactions")
        response = review_answer(question, response, review)

    return response

if __name__ == "__main__":
    result = reflection_agent(
        "Explain the concept of RAG and give me some examples, but remember I'm a child"
    )
    print(result)