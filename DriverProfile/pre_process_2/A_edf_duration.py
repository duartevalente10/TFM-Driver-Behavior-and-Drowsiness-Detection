import os
import pandas as pd
import pyedflib

# Diretório com os ficheiros EDF
edf_directory = '../datasets_2/valu3s/vitaport/'

# Listar todos os ficheiros .edf no diretório
edf_files = [f for f in os.listdir(edf_directory) if f.endswith('.edf')]

# Criar listas para armazenar os nomes e as durações
file_names = []
durations = []

# Iterar sobre os ficheiros e calcular a duração
for edf_file in edf_files:
    edf_path = os.path.join(edf_directory, edf_file)
    with pyedflib.EdfReader(edf_path) as f:
        duration = f.file_duration  # duração em segundos
        file_names.append(edf_file)
        durations.append(duration)

# Criar o DataFrame
df = pd.DataFrame({
    'Filename': file_names,
    'Duration': durations
})

# Exibir o DataFrame
print(df)

# save
df.to_csv('datasets/hrv/hrv_duration.csv', index=False)