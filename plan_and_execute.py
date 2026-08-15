import os
import json
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

cliente = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# --- Modelos de Domínio (Pydantic) ---

class Etapa(BaseModel):
    id: int
    descricao: str
    ferramenta: Optional[str] = None
    depende_de: List[int] = Field(default_factory=list)


class Plano(BaseModel):
    objetivo: str
    etapas: List[Etapa]


# --- Componentes Principais ---

def gerar_plano(objetivo: str, ferramentas_disponiveis: List[str]) -> Plano:
    """Planejador: Decompõe um objetivo em etapas executáveis e dependências."""
    prompt = f"""
    Decomponha o objetivo abaixo em etapas executáveis sequenciais.
    Use APENAS as ferramentas listadas. Identifique as dependências entre as etapas.

    Objetivo: {objetivo}
    Ferramentas disponíveis: {ferramentas_disponiveis}

    Retorne um JSON que corresponda ao esquema:
    {{
        "objetivo": "...",
        "etapas": [
            {{"id": 1, "descricao": "...", "ferramenta": "...", "depende_de": []}}
        ]
    }}
    """
    resposta = cliente.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return Plano(**json.loads(resposta.choices[0].message.content or "{}"))


def executar_etapa(
    etapa: Etapa, contexto: Dict[int, Any], ferramentas: Dict[str, Callable]
) -> str:
    """Executor: Executa uma etapa individual usando uma ferramenta determinística ou raciocínio de LLM."""
    if etapa.ferramenta and etapa.ferramenta in ferramentas:
        # Execução de ferramenta determinística
        resultado = ferramentas[etapa.ferramenta](etapa.descricao, contexto)
    else:
        # Execução de fallback com LLM quando nenhuma ferramenta é atribuída
        prompt = f"""
        Execute a etapa abaixo usando o contexto das etapas anteriores.
        Etapa: {etapa.descricao}
        Contexto acumulado: {contexto}
        """
        resposta = cliente.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
        )
        resultado = resposta.choices[0].message.content or ""

    return str(resultado)


def executar_plano(plano: Plano, ferramentas: Dict[str, Callable]) -> Dict[int, str]:
    """Orquestrador: Executa as etapas resolvendo a ordem de dependência (resolução topológica)."""
    resultados: Dict[int, str] = {}
    etapas_pendentes = list(plano.etapas)

    while etapas_pendentes:
        # Encontra etapas cujas dependências estão totalmente resolvidas
        etapas_prontas = [
            etapa
            for etapa in etapas_pendentes
            if all(dep in resultados for dep in etapa.depende_de)
        ]

        if not etapas_prontas:
            raise RuntimeError("Dependência circular detectada no grafo de execução do plano.")

        for etapa in etapas_prontas:
            print(f"Executando etapa {etapa.id}: {etapa.descricao}")
            contexto_etapa = {dep: resultados[dep] for dep in etapa.depende_de}
            resultados[etapa.id] = executar_etapa(etapa, contexto_etapa, ferramentas)
            etapas_pendentes.remove(etapa)

    return resultados


# --- Implementações das Ferramentas ---

def calculadora_segura(expressao: str, _contexto: dict) -> str:
    """Avalia com segurança expressões aritméticas básicas."""
    caracteres_permitidos = set("0123456789+-*/(). ")
    if not all(char in caracteres_permitidos for char in expressao):
        return "Erro: A expressão contém caracteres proibidos."
    try:
        return str(eval(expressao, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Erro de cálculo: {exc}"


# --- Ponto de Entrada ---

if __name__ == "__main__":
    ferramentas = {
        "pesquisa_web": lambda q, ctx: "...resultados da pesquisa...",
        "calcular": calculadora_segura,
        "consultar_bd": lambda q, ctx: "...registros do banco de dados...",
    }

    plano = gerar_plano(
        objetivo="Analisar as vendas do terceiro trimestre e gerar um relatório executivo",
        ferramentas_disponiveis=list(ferramentas.keys()),
    )

    resultados_execucao = executar_plano(plano, ferramentas)
    print("\nExecução Concluída:")
    print(json.dumps(resultados_execucao, indent=2))