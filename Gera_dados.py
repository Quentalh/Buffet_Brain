import torch
import torch.nn as nn
import numpy as np
import json

# 1. Recriar a Rede (para poder carregar os pesos)
class Guia_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Guia_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

print("A iniciar o Processamento em Lote (Batch) da IA...")

# 2. Carregar o Cérebro
checkpoint = torch.load('lstm_guide.pth', map_location=torch.device('cpu'))
modelo = Guia_LSTM(5, checkpoint['hidden_size'], 3)
modelo.load_state_dict(checkpoint['model_state_dict'])
modelo.eval()

X_mean = checkpoint['X_mean'].numpy()
X_std = checkpoint['X_std'].numpy()
mapeamento = checkpoint['mapeamento_reverso']

# 3. Criar os nossos 3 Clientes Artificiais (com históricos diferentes)
# Formato do histórico: [idade, renda, poupanca, conhecimento, risco] ao longo de 4 anos
clientes_artificiais = [
    {
        "nome": "João Conservador", "email": "joao@teste.com", "senha": "123",
        "renda_atual": 4000, "poupanca_atual": 0.10,
        "historico": [
            [25, 3000, 0.05, 1, 1], [26, 3200, 0.05, 1, 1], [27, 3500, 0.08, 2, 1], [28, 4000, 0.10, 2, 1]
        ]
    },
    {
        "nome": "Maria Moderada", "email": "maria@teste.com", "senha": "123",
        "renda_atual": 8000, "poupanca_atual": 0.20,
        "historico": [
            [30, 6000, 0.10, 2, 2], [31, 6500, 0.15, 3, 2], [32, 7000, 0.15, 3, 3], [33, 8000, 0.20, 3, 3]
        ]
    },
    {
        "nome": "Carlos Audacioso", "email": "carlos@teste.com", "senha": "123",
        "renda_atual": 15000, "poupanca_atual": 0.35,
        "historico": [
            [22, 5000, 0.20, 3, 4], [23, 8000, 0.25, 4, 4], [24, 12000, 0.30, 5, 5], [25, 15000, 0.35, 5, 5]
        ]
    }
]

# 4. Passar todos pela LSTM e guardar no Banco
banco_de_dados = {}

with torch.no_grad():
    for cliente in clientes_artificiais:
        hist_np = np.array([cliente['historico']])
        hist_scaled = (hist_np - X_mean) / X_std
        tensor = torch.FloatTensor(hist_scaled)
        
        output = modelo(tensor)
        _, predicted = torch.max(output.data, 1)
        perfil_ia = mapeamento[predicted.item()]
        
        # Gravar no "Banco de Dados" usando o email como chave
        banco_de_dados[cliente['email']] = {
            "nome": cliente['nome'],
            "senha": cliente['senha'], # Na vida real seria criptografada!
            "renda_atual": cliente['renda_atual'],
            "poupanca_atual": cliente['poupanca_atual'],
            "perfil_classificado_pela_ia": perfil_ia
        }

# Guardar ficheiro JSON
with open('usuarios.json', 'w', encoding='utf-8') as f:
    json.dump(banco_de_dados, f, ensure_ascii=False, indent=4)

print("✅ Base de dados 'usuarios.json' gerada com sucesso! A IA já classificou todos.")