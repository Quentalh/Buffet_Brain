import torch
import torch.nn as nn
import numpy as np
import json

# Definição da arquitetura para compatibilidade com o Checkpoint
class Guia_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Guia_LSTM, self).__init__()
        # Se usar num_layers=2, o dropout interno da LSTM é ativado
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.3, batch_first=True)
        self.dropout_fc = nn.Dropout(0.4) # Desliga 40% das conexões antes da decisão final
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Pega o último passo temporal e aplica o dropout
        out = self.dropout_fc(lstm_out[:, -1, :])
        return self.fc(out)

print("A iniciar o Processamento em Lote (Batch) da IA...")

# Carregando os metadados do treinamento
checkpoint = torch.load('lstm_guide.pth', map_location=torch.device('cpu'), weights_only=False)
modelo = Guia_LSTM(2, checkpoint['hidden_size'], 3)
modelo.load_state_dict(checkpoint['model_state_dict'])
modelo.eval()

X_mean = checkpoint['X_mean']
X_std = checkpoint['X_std']
mapeamento = checkpoint['mapeamento_reverso']

# Fixando seed local para estabilidade dos dados dos clientes de teste
np.random.seed(42)

# Gerando históricos complexos de 120 meses para simulação em produção
historico_joao = []
renda_j = 4000
for m in range(120):
    renda_j += np.random.normal(15, 50)
    poupanca_j = np.random.normal(0.08, 0.02)
    historico_joao.append([round(max(2000, renda_j), 2), round(max(0.01, min(poupanca_j, 0.8)), 4)])

historico_maria = []
renda_m = 7500
for m in range(120):
    renda_m += (np.sin(m / 3.0) * 150) + np.random.normal(20, 80)
    poupanca_m = np.random.normal(0.18, 0.04)
    historico_maria.append([round(max(3000, renda_m), 2), round(max(0.01, min(poupanca_m, 0.8)), 4)])

historico_carlos = []
renda_c = 12000
for m in range(120):
    renda_c += np.random.normal(80, 300) + np.random.choice([-300, 300], p=[0.3, 0.7])
    poupanca_c = np.random.normal(0.32, 0.08)
    historico_carlos.append([round(max(4000, renda_c), 2), round(max(0.01, min(poupanca_c, 0.8)), 4)])

clientes_artificiais = [
    {
        "nome": "João Conservador", "email": "joao@teste.com", "senha": "123",
        "renda_atual": round(historico_joao[-1][0], 2), "poupanca_atual": round(historico_joao[-1][1], 4),
        "historico": historico_joao
    },
    {
        "nome": "Maria Moderada", "email": "maria@teste.com", "senha": "123",
        "renda_atual": round(historico_maria[-1][0], 2), "poupanca_atual": round(historico_maria[-1][1], 4),
        "historico": historico_maria
    },
    {
        "nome": "Carlos Audacioso", "email": "carlos@teste.com", "senha": "123",
        "renda_atual": round(historico_carlos[-1][0], 2), "poupanca_atual": round(historico_carlos[-1][1], 4),
        "historico": historico_carlos
    }
]

banco_de_dados = {}

with torch.no_grad():
    for cliente in clientes_artificiais:
        # Formata o histórico individual para a dimensão (1, 120, 2) esperada pela LSTM
        hist_np = np.array([cliente['historico']])
        hist_scaled = (hist_np - X_mean) / X_std
        tensor = torch.FloatTensor(hist_scaled)
        
        output = modelo(tensor)
        _, predicted = torch.max(output.data, 1)
        perfil_ia = mapeamento[str(predicted.item()) if str(predicted.item()) in mapeamento else predicted.item()]
        
        banco_de_dados[cliente['email']] = {
            "nome": cliente['nome'],
            "senha": cliente['senha'],
            "renda_atual": cliente['renda_atual'],
            "poupanca_atual": cliente['poupanca_atual'],
            "perfil_classificado_pela_ia": perfil_ia,
            "historico": cliente['historico']
        }

with open('usuarios.json', 'w', encoding='utf-8') as f:
    json.dump(banco_de_dados, f, ensure_ascii=False, indent=4)

print("✅ Base de dados 'usuarios.json' gerada com sucesso! A IA classificou a série temporal de 120 meses de todos.")