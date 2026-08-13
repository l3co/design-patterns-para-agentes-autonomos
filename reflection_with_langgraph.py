from typing import TypedDict
from langgraph.graph import StateGraph, END

from reflection import (
    generate_initial_response,
    evaluate_response,
    refine_response
)


class ReflectionState(TypedDict):
    question: str
    answer: str
    reviews: list[dict]
    iterations: int
    approved: bool


def generate_node(state: ReflectionState) -> dict:
    # Initial generation on the first pass
    if state["iterations"] == 0:
        answer = generate_initial_response(state["question"])
    # Refine the existing response using the latest review
    else:
        last_review = state["reviews"][-1]
        answer = refine_response(
            state["question"],
            state["answer"],
            last_review
        )

    # Return key 'answer' to match the ReflectionState schema
    return {
        "answer": answer,
        "iterations": state["iterations"] + 1
    }


def evaluate_node(state: ReflectionState) -> dict:
    review = evaluate_response(
        state["question"],
        state["answer"]
    )

    # Append new review and update the approved flag
    return {
        "reviews": state["reviews"] + [review],
        "approved": review.get("approved", False)
    }


def should_continue(state: ReflectionState) -> str:
    # Stop condition: approved or reached maximum iterations
    if state["approved"] or state["iterations"] >= 3:
        return END
    # Route back to the generator node
    return "generator"


# Graph Definition
graph = StateGraph(ReflectionState)

graph.add_node("generator", generate_node)
graph.add_node("evaluator", evaluate_node)

graph.set_entry_point("generator")

graph.add_edge("generator", "evaluator")
graph.add_conditional_edges("evaluator", should_continue)

app = graph.compile()


if __name__ == "__main__":
    initial_state: ReflectionState = {
        "question": "Explain the concept of RAG and give me some examples, but remember I'm a child",
        "answer": "",
        "reviews": [],
        "iterations": 0,
        "approved": False
    }

    result = app.invoke(initial_state)

    print(f"Final Approval Status: {result['approved']}")
    print(f"Total Iterations: {result['iterations']}")
    print("\nFinal Answer:\n", result["answer"])