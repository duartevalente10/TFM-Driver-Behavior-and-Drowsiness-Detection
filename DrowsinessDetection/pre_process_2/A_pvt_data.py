import pandas as pd

"""
Psychomotor Vigilance Task para avaliar o tempo de reação e a consistência com que os participantes respondem a estímulos visuais.

Features:

PVT_Trials.thisRepN > ID da repetição 
PVT_Trials.thisTrialN > é sempre 0 em todos os trails
PVT_Trials.thisN > ID igual ao thisRepN 
PVT_Trials.thisIndex > é sempre 0 em todos os trails 
key_resp.keys > Tecla pressionada pelo participante
key_resp.rt > Tempo de reação
key_resp.started
key_resp.stopped
ISI > Intervalo inter-estímulo  entre os ensaios
dontresp.keys
Accuracy > Precisão das respostas
RTms > Tempo de reação em milissegundos
Response.keys > Tecla pressionada na resposta ao estímulo
Response.rt > Tempo de reação de resposta ao estímulo sem segundos
Response.started
Response.stopped
ID > ID do participante Exemplo: "f19_3_before"
Tid > ID do trail do participante "3" no caso do exemplo
KSS > Escala de Sonolência (1 a 9)
date > Data no formato Ano_Mês_Dia_0219
expName > Nome do teste que é sempre PVT 
psychopyVersion > Versão do Psychopy utilizada
frameRate > framerate do teste que foi sempre iagual a 60.055839911674546 hz

"""

# load the CSV file
file_path = '../datasets_2/valu3s/PVT/f19_3_before_PVT_2023_Mar_18_0029.csv'
data = pd.read_csv(file_path, delimiter=';')

# number of instances 
num_instances = data.shape[0]

# number of features 
num_features = data.shape[1]

# feature names
feature_names = data.columns.tolist()

# basic statistics for numerical features
basic_stats = data.describe()

# output the results
print("\nNumber of instances:", num_instances)
print("Number of features:", num_features)
print("Feature names:", feature_names)
print("\nBasic statistics:")
print(basic_stats)
