"""
Arquivo de configuração.
"""

from pathlib import Path
import json

CONFIG_FILE = "config.json"

c_training_data_path = Path("./training_data")  # onde os dados de treinamento estão localizados
c_testing_data_path = Path("./testing_data") 
c_epsilon = 0.5
c_threshold = 0.1  # O treinamento acaba quando o erro for menor que o threshold
c_sleep = 1  # No loop principal quanto tempo a thread precisa dormir, a cada geração

def store_config():
    """Salva todas as variáveis de configuração em um arquivo JSON."""

    config = {}

    for name, value in globals().items():
        if not name.startswith("c_"):
            continue

        if isinstance(value, Path):
            config[name] = str(value)
        else:
            config[name] = value

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def load_config():
    """Carrega as variáveis de configuração do arquivo JSON."""

    global_vars = globals()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        for name, value in config.items():
            if name not in global_vars:
                continue

            current_value = global_vars[name]

            # Restaurar tipos especiais
            if isinstance(current_value, Path):
                global_vars[name] = Path(value)
            else:
                global_vars[name] = value
    except:
        ...

load_config()