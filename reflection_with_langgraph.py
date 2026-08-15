from typing import TypedDict
from langgraph.graph import StateGraph, END

from reflection import (
    gerar_resposta_inicial,
    avaliar_resposta,
    refinar_resposta
)


class EstadoReflexao(TypedDict):
    pergunta: str
    resposta: str
    avaliacoes: list[dict]
    iteracoes: int
    aprovado: bool


def no_gerador(state: EstadoReflexao) -> dict:
    # Geração inicial na primeira passagem
    if state["iteracoes"] == 0:
        resposta = gerar_resposta_inicial(state["pergunta"])
    # Refinar a resposta existente usando a última avaliação
    else:
        ultima_avaliacao = state["avaliacoes"][-1]
        resposta = refinar_resposta(
            state["pergunta"],
            state["resposta"],
            ultima_avaliacao
        )

    # Retornar a chave 'resposta' para corresponder ao esquema EstadoReflexao
    return {
        "resposta": resposta,
        "iteracoes": state["iteracoes"] + 1
    }


def no_avaliador(state: EstadoReflexao) -> dict:
    avaliacao = avaliar_resposta(
        state["pergunta"],
        state["resposta"]
    )

    # Anexar nova avaliação e atualizar a flag de aprovação
    return {
        "avaliacoes": state["avaliacoes"] + [avaliacao],
        "aprovado": avaliacao.get("aprovado", False)
    }


def deve_continuar(state: EstadoReflexao) -> str:
    # Condição de parada: aprovado ou atingiu o número máximo de iterações
    if state["aprovado"] or state["iteracoes"] >= 3:
        return END
    # Rota de volta para o nó gerador
    return "gerador"


# Definição do Grafo
grafo = StateGraph(EstadoReflexao)

grafo.add_node("gerador", no_gerador)
grafo.add_node("avaliador", no_avaliador)

grafo.set_entry_point("gerador")

grafo.add_edge("gerador", "avaliador")
grafo.add_conditional_edges("avaliador", deve_continuar)

app = grafo.compile()


if __name__ == "__main__":
    estado_inicial: EstadoReflexao = {
        "pergunta": "Explique o conceito de RAG e me dê alguns exemplos, mas lembre-se que eu sou uma criança",
        "resposta": "",
        "avaliacoes": [],
        "iteracoes": 0,
        "aprovado": False
    }

    resultado = app.invoke(estado_inicial)

    print(f"Status Final de Aprovação: {resultado['aprovado']}")
    print(f"Total de Iterações: {resultado['iteracoes']}")
    print("\nResposta Final:\n", resultado["resposta"])