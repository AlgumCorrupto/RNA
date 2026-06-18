"""
Arquivo principal, onde se localiza o código da rede neural.

Cada função step_%d corresponde a um passo a passo do algoritmo de treinamento.

Com exceção da função step_5_6 em que eu mesclei os dois passos em uma única
pois fazia mas sentido.
"""


import numpy as np
from numpy.typing import NDArray
import config
import os
import random
import asyncio

def nothing():
    """
    O motivo dessa função existir é um artefato da interface gráfica.
    Na versão com interface gráfica. Esse é um callback de on_next_gen.
    """
    ...

class training_data:
    """
    Os dados de treinamento ficam armazenados em arquivos de texto separados
    a função load_data ler esses arquivos e carregam eles como uma 
    lista de training_data
    """
    def __init__(self, x: NDArray[np.float64], y: tuple[float, float]):
        self.x = x
        self.y = y

# a lista que armazena os dados de treinamento seguido para o índice do dado
# sendo utilizado atualmente.
data: list[training_data] = []
curr = 0;

# A cada geração on_next_gen é chamado, na versão terminal um callback que faz nada.
# Porém na versão com GUI, um callback especial é chamado mandando
# um evento para o resto da aplicação que ocorreu uma mudança de geração

on_next_gen = nothing

# lista contendo os erros de todas as gerações
errors: list[float] = []
weights: tuple[NDArray[np.float64], NDArray[np.float64]] # as sinapsaes

# função responsável por ler os dados armazenados em arquivos
def load_data():
    global data
    data = []
    files = os.listdir(config.DATA_PATH)
    for file in files:
        with open(os.path.join(config.DATA_PATH, file), "r") as buffer:
            try:
                y_str = buffer.readline().strip()
                y = eval(y_str)
                x_str = buffer.read()
                x_str = x_str.strip()

                # eu fiz um truque aqui para não explicitamente fazer o parse da matriz.
                # a função np.matrix pode aceitar uma string exatamente no mesmo formato
                # como foi declarado no arquivo.

                # nessa linha eu só estou removendo o último ;
                # se eu não tiver removido, o construtor teria interpretado uma matriz 
                # de 3 linhas tendo 4 linhas.
                if x_str.endswith(";"):
                    x_str = x_str[:-1]

                m = np.matrix(x_str, dtype=np.float64)
                x = np.concatenate((np.array([1.0], np.float64), np.asarray(m, dtype=np.float64).flatten()))
                data.append(training_data(x, y))
            except:
                print(f"Arquivo {file} inválido, portanto ele não foi carregado")

    random.shuffle(data) # aleatorizando as posições


def activate(input: float) -> float:
    """
    Função de ativação threshold
    """
    if input >= 0:
        return 1
    else:
        return -1


def mean(values):
    """
    Computa a média de uma lista
    """
    return sum(values) / len(values);

"""
Aqui está o passo a passo, mapeados para as etapas de treinamento
vistas em aula.
"""

def step_1() -> None:
    global weights
    rnd = np.random.default_rng()
    weights = (rnd.random(10, np.float64) * 2 - 1.0, rnd.random(10, np.float64)* 2 - 1.0);

def step_2() -> tuple[float, float]:
    global weights, curr, data, errors

    print("Geração", len(errors))
    inputs = data[curr].x

    first = np.dot(inputs, weights[0])
    second = np.dot(inputs, weights[1])

    return (first, second)

def step_3(raw: tuple[float, float]) -> tuple[float, float]:
    in_first, in_second = raw
    return (activate(in_first), activate(in_second))

def step_4(output: tuple[float, float]) -> tuple[float, float]:
    global data, curr
    o_1, o_2 = output
    d_1, d_2 = data[curr].y
    result = (d_1 - o_1, d_2 - o_2)
    print("resultado esperado:", (d_1, d_2))
    print("resultado calculado:", (o_1, o_2))
    print("erro:", result)
    return result

def step_5_6(error: tuple[float, float]) -> None:
    global data, curr, weights
    e_1, e_2 = error
    delta_1 = e_1 * config.EPSILON * data[curr].x
    delta_2 = e_2 * config.EPSILON * data[curr].x

    w1, w2 = weights
    weights = (delta_1 + w1, delta_2 + w2)

    errors.append(((e_1 ** 2) + (e_2 ** 2)) / 2)

    curr += 1
    curr %= len(data)

load_data()

async def train():
    """
    Loop de treinamento principal,
    o motivo dessa função ser assíncrona é
    que a interface gráfica e o código de treinamento
    devem ser rodados em threads diferentes para a plotagem
    do gráfico em tempo real.
    """

    global on_next_gen, errors, curr, data
    errors = []
    curr = 0
    step_1()
    while True:
        raw = step_2()
        activated = step_3(raw)
        err = step_4(activated)
        step_5_6(err)

        if curr == 0:
            epoch_mean = mean(errors[-len(data):]) # pegando os erros da época
            print(f"média da época {len(errors) // len(data)}: {epoch_mean}")
            random.shuffle(data)
        
            if epoch_mean < config.THRESHOLD:
                break
        print('')
        on_next_gen()

        if(config.SLEEP > 0):
            await asyncio.sleep(config.SLEEP)

if __name__ == "__main__":
    asyncio.run(train())

