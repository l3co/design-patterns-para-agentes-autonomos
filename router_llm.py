import json
import os

from openai import OpenAI

cliente = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

ROTAS = {
    "modelo_pequeno": {"modelo": "gpt-5-mini", "descricao": "Perguntas factuais simples, resumos curtos, formatação.",
                       "custo_relativo": 1, },
    "modelo_grande": {"modelo": "gpt-5", "descricao": "Raciocínio complexo, análise técnica, código difícil.",
                      "custo_relativo": 15, },
    "modelo_codigo": {"modelo": "gpt-5", "descricao": "Geração e refatoração de código em Python.",
                      "custo_relativo": 10, },
    "rag": {"modelo": "pipeline-rag", "descricao": "Perguntas sobre documentos internos da empresa.",
            "custo_relativo": 5, },
}


def pipeline_rag(pergunta: str) -> str:
    return f"Resposta via RAG para : {pergunta}"


def router_llm(pergunta: str) -> str:
    descricao_rotas = "\n".join(
        f"-{nome}: {meta['descricao']}" for nome, meta in ROTAS.items()
    )

    prompt = f"""
    Analise a pergunta abaixo e decida a melhor rota.
    
    Rotas disponiveis: 
    {descricao_rotas}
    
    Pergunta: {pergunta}
    
    Return JSON: {{"rota": "nome_da_rota", "razao": "..." }}
    """

    resp = cliente.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )

    rota = json.loads(resp.choices[0].message.content or "{}").get("rota")
    if rota not in ROTAS:
        return "modelo_pequeno"

    return rota


def executar_com_router(pergunta: str) -> str:
    rota = router_llm(pergunta)
    print(f"Pergunta roteada para : {rota}")

    if rota == "rag":
        return pipeline_rag(pergunta)

    modelo = ROTAS[rota]["modelo"]
    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": pergunta}]
    )

    return resp.choices[0].message.content or ""


if __name__ == '__main__':
    print(executar_com_router("Qual a capital da França?"))
    print(executar_com_router("Refatore esta classe Python para usar async/await"))
    print(executar_com_router("Quanto faturamos no Q3 segundo nosso ERP?"))
