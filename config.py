"""
Arquivo de configuração.
"""

from pathlib import Path

DATA_PATH=Path("./data") # onde os dados estão localizados
EPSILON=0.5
THRESHOLD=0.1 # O treinamento acaba quando o erro for menor que o threshold
SLEEP=1 # No loop principal quanto tempo a thread precisa dormir, a cada geração
