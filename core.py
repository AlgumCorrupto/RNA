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
from pathlib import Path

def nothing():
    """
    O motivo dessa função existir é um artefato da interface gráfica.
    Na versão com interface gráfica. Esse é um callback de on_next_gen.
    """
    ...

class manuscript_image:
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
training_data: list[manuscript_image] = []

testing_data: list[manuscript_image] = []

# A cada geração on_next_gen é chamado, na versão terminal um callback que faz nada.
# Porém na versão com GUI, um callback especial é chamado mandando
# um evento para o resto da aplicação que ocorreu uma mudança de geração

on_next_gen = nothing

# lista contendo os erros de todas as gerações
errors: list[float] = []
weights: tuple[NDArray[np.float64], NDArray[np.float64]] # as sinapsaes

# função responsável por ler os dados armazenados em arquivos
def load_images(path: Path):
    data = []
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file), "r") as buffer:
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
                data.append(manuscript_image(x, y))
            except:
                print(f"Arquivo {file} inválido, portanto ele não foi carregado")

    return data

def load_training_data():
    global training_data
    training_data = load_images(config.c_training_data_path)
    random.shuffle(training_data)

def load_testing_data():
    global testing_data
    testing_data = load_images(config.c_testing_data_path)
    random.shuffle(testing_data)


def threshold(input: float) -> float:
    """
    Função de ativação threshold
    """
    if input >= 0:
        return 1
    else:
        return -1

activate = threshold


def mean(values):
    """
    Computa a média de uma lista
    """
    return sum(values) / len(values);

"""
Aqui está o passo a passo, mapeados para as etapas de treinamento
vistas em aula.
"""

def init_random_weights() -> None:
    global weights
    rnd = np.random.default_rng()
    weights = (rnd.random(10, np.float64) * 2 - 1.0, rnd.random(10, np.float64)* 2 - 1.0);

def calculate_raw_output(data: manuscript_image) -> tuple[float, float]:
    global weights, errors

    inputs = data.x

    first = np.dot(inputs, weights[0])
    second = np.dot(inputs, weights[1])

    return (first, second)

def activate_output(raw: tuple[float, float]) -> tuple[float, float]:
    in_first, in_second = raw
    return (threshold(in_first), threshold(in_second))

def calculate_errors(output: tuple[float, float], data: manuscript_image) -> tuple[float, float]:
    o_1, o_2 = output
    d_1, d_2 = data.y
    result = (d_1 - o_1, d_2 - o_2)
    print("resultado esperado:", (d_1, d_2))
    print("resultado calculado:", (o_1, o_2))
    print("erro:", result)
    return result

def update_weights_and_store_errors(error: tuple[float, float], data: manuscript_image) -> None:
    global weights
    e_1, e_2 = error
    delta_1 = e_1 * config.c_epsilon * data.x
    delta_2 = e_2 * config.c_epsilon * data.x

    w1, w2 = weights
    weights = (delta_1 + w1, delta_2 + w2)

    errors.append(((e_1 ** 2) + (e_2 ** 2)) / 2)


def test_value(data: manuscript_image):
    raw = calculate_raw_output(data)
    activated = activate_output(raw)
    return calculate_errors(activated, data)

async def test_working_data():
    global testing_data
    print('Testando...')
    try:
        for data in testing_data:
            test_value(data)
            print('')
    except:
        print("Sinapses não carregadas, treine a rede neural primeiro!")
    print('Fim dos testes.\n')

load_training_data()
load_testing_data()

async def train():
    """
    Loop de treinamento principal,
    o motivo dessa função ser assíncrona é
    que a interface gráfica e o código de treinamento
    devem ser rodados em threads diferentes para a plotagem
    do gráfico em tempo real.
    """
    global on_next_gen, errors, training_data
    print("Treinando...")
    errors = []
    training_data_i = 0
    init_random_weights()
    while True:
        print("Geração", len(errors))
        data = training_data[training_data_i]
        err = test_value(data)
        update_weights_and_store_errors(err, data)

        training_data_i += 1
        training_data_i %= len(training_data)

        if training_data_i == 0:
            epoch_mean = mean(errors[-len(training_data):]) # pegando os erros da época
            print(f"média da época {len(errors) // len(training_data)}: {epoch_mean}")
            random.shuffle(training_data)
            print('')
        
            if epoch_mean < config.c_threshold:
                break
        on_next_gen()

        print('')

        if(config.c_sleep > 0):
            await asyncio.sleep(config.c_sleep)
    print('Fim do treinamento.\n')

if __name__ == "__main__":
    asyncio.run(train())
    asyncio.run(test_working_data())
