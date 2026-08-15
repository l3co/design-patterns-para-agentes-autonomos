import json
import os
from openai import OpenAI

cliente = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def gerar_resposta_inicial(pergunta: str) -> str:
    resposta = cliente.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "Você é um analista de dados sênior"},
            {"role": "user", "content": pergunta},
        ],
    )
    return resposta.choices[0].message.content or ""


def avaliar_resposta(pergunta: str, resposta_atual: str) -> dict:
    prompt_avaliacao = f"""
    Você é um revisor técnico sênior. Analise a resposta abaixo:

    Pergunta Original: {pergunta}
    Resposta Atual: {resposta_atual}

    Retorne em JSON: 
    - "aprovado" (true/false)
    - "problemas" (uma lista de preocupações sobre a resposta)
    - "sugestoes" (uma lista de sugestões para melhorar a resposta)
    """

    resposta = cliente.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt_avaliacao}],
        response_format={"type": "json_object"},
    )

    return json.loads(resposta.choices[0].message.content or "{}")


def refinar_resposta(pergunta: str, resposta_atual: str, avaliacao: dict) -> str:
    prompt_refinamento = f"""
    Alguns problemas foram levantados em relação à sua resposta. Reescreva-a, corrigindo todos os problemas apontados. 

    Pergunta Original: {pergunta}    
    Resposta Atual: {resposta_atual}
    Problemas: {avaliacao.get('problemas', [])}
    Sugestões: {avaliacao.get('sugestoes', [])}
    """

    resposta = cliente.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt_refinamento}],
    )

    return resposta.choices[0].message.content or ""


def executar_agente_reflexao(pergunta: str, max_iteracoes: int = 3) -> str:
    resposta_atual = gerar_resposta_inicial(pergunta)

    for iteracao in range(max_iteracoes):
        avaliacao = avaliar_resposta(pergunta, resposta_atual)

        if avaliacao.get("aprovado"):
            print(f"Aprovado na iteração {iteracao + 1}")
            return resposta_atual

        print(f"Refinando na iteração {iteracao + 1}")
        resposta_atual = refinar_resposta(pergunta, resposta_atual, avaliacao)

    return resposta_atual


if __name__ == "__main__":
    resultado = executar_agente_reflexao(
        "Explique o conceito de RAG e me dê alguns exemplos, mas lembre-se que eu sou uma criança"
    )
    print(resultado)