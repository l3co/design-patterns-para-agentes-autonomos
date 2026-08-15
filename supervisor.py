import os
from unittest import result

from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def worker_pesquisa(tarefa: str) -> str :
    """Worker que faz pesquisa de informação"""
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Você é pesquisador. Busque informações relevantes e factuais."},
            {"role": "user", "content": tarefa},
        ],
    )
    return resp.choices[0].message.content or ""

def worker_analise(tarefa: str) -> str :
    """Worker que faz análise técnica"""
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Você é um analista Senior. Interprete dados com rigor tecnico"},
            {"role": "user", "content": tarefa},
        ]
    )

    return resp.choices[0].message.content or ""

def worker_redacao(tarefa: str) -> str :
    """Worker que produz texto final em formato apropriado"""
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Você é um redator técnico. Produza texto claro e bem estruturado"},
            {"role": "user", "content": tarefa}
        ]
    )

    return resp.choices[0].message.content or ""


WORKERS = {
    "pesquisa": worker_pesquisa,
    "analise": worker_analise,
    "redacao" : worker_redacao
}

def supervisor(objetivo: str, historico: list) -> dict:
    prompt = f"""
    Você é o supervisor de uma equipe de agentes.
    Workers disponiveis: pesquisa, analise, redacao
    
    Objetivo: {objetivo}
    Historico de execução : 
    {json.dumps(historico, ensure_ascii=False, indent=2)}
    
    Retorne JSON : 
    - "proximo_worker": "pesquisa" | "analise" | "redacao" | "FIM"
    - "tarefa": instrução clara para o worker (vazio se FIM)
    - "justificativa": por que escolheu este worker
    """

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(resp.choices[0].message.content or "{}")

# Sistema de orquestração
def sistema_supervisionado(objetivo: str, max_passos: int = 8) -> list:
    historico = []
    for passo in range(max_passos):
        decisao = supervisor(objetivo, historico)
        print(f"Passo {passo + 1}: {decisao['justificativa']}")
        if decisao["proximo_worker"] == "FIM":
            break

        worker_fn = WORKERS.get(decisao["proximo_worker"])
        if worker_fn is None:
            raise ValueError(f"Worker inválido: {decisao['proximo_worker']}")

        resultado = worker_fn(decisao["tarefa"])
        historico.append({
            "worker": decisao["proximo_worker"],
            "tarefa": decisao["tarefa"],
            "resultado": resultado
        })
    return historico

if __name__ == '__main__':
    resultado = sistema_supervisionado(
        "Produzir um briefing executivo sobre tendencias em LLMs em 2026"
    )

    for passo in resultado:
        print(f"\n[{passo['worker']}] {passo['resultado'][:200]}...")