import json
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def generate_initial_response(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "You are a senior data analyst"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


def evaluate_response(question: str, current_response: str) -> dict:
    evaluation_prompt = f"""
    You are a senior technical reviewer. Analyze the answer below:

    Original Question: {question}
    Current Response: {current_response}

    Return in JSON: 
    - "approved" (true/false)
    - "problems" (a list of concerns about the answer)
    - "suggestions" (a list of suggestions to improve the answer)
    """

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": evaluation_prompt}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content or "{}")


def refine_response(question: str, current_response: str, evaluation: dict) -> str:
    refinement_prompt = f"""
    Some issues were raised regarding your response. Rewrite it, correcting all the issues pointed out. 

    Original Question: {question}    
    Current Response: {current_response}
    Problems: {evaluation.get('problems', [])}
    Suggestions: {evaluation.get('suggestions', [])}
    """

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": refinement_prompt}],
    )

    return response.choices[0].message.content or ""


def run_reflection_agent(question: str, max_iterations: int = 3) -> str:
    current_response = generate_initial_response(question)

    for iteration in range(max_iterations):
        evaluation = evaluate_response(question, current_response)

        if evaluation.get("approved"):
            print(f"Approved at iteration {iteration + 1}")
            return current_response

        print(f"Refining at iteration {iteration + 1}")
        current_response = refine_response(question, current_response, evaluation)

    return current_response


if __name__ == "__main__":
    result = run_reflection_agent(
        "Explain the concept of RAG and give me some examples, but remember I'm a child"
    )
    print(result)