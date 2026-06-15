from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from google import genai
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

# ==============================================================================
# CONFIGURAÇÃO DE INFRAESTRUTURA DE SERVIDOR
# ==============================================================================
app = Flask(__name__)
CORS(app) 

# Inicialização do cliente API para o ambiente generativo
# CHAVE REMOVIDA PARA O GITHUB NÃO BLOQUEAR O PUSH
client = genai.Client(api_key="COLOQUE_SUA_CHAVE_AQUI_APENAS_NA_HORA_DE_RODAR")

# ==============================================================================
# MOTOR QUANTITATIVO (Filtros Financeiros e Integração de Ativos da B3)
# ==============================================================================
def buscar_melhores_acoes(perfil):
    print("Iniciando varredura quantitativa via API Global (Yahoo Finance)...")
    
    # POOL DE ATIVOS (Universo de Busca Restrito / Gestão de Riscos de Liquidez)
    pool_ativos = [
            # Setor Elétrico / Saneamento / Seguros (Conservadores - Foco em DY)
            'TAEE11.SA', 'EGIE3.SA', 'EQTL3.SA', 'TRPL4.SA', 'SBSP3.SA', # SBSP3 substituiu a CPLE6
            'ENBR3.SA', 'BBSE3.SA', 'CXSE3.SA', 'SANB11.SA', 'CMIG4.SA',
            
            # Bancos, Commodities e Grandes Corporações (Moderados - Foco em ROE)
            'ITUB4.SA', 'BBAS3.SA', 'BBDC4.SA', 'PETR4.SA', 'VALE3.SA', 
            'SUZB3.SA', 'KLBN11.SA', 'ABEV3.SA', 'JBSS3.SA', 'B3SA3.SA',
            
            # Tecnologia, Varejo, Saúde e Expansão (Audaciosos - Foco em Crescimento)
            'WEGE3.SA', 'RADL3.SA', 'TOTS3.SA', 'RENT3.SA', 'VIVT3.SA', 
            'LREN3.SA', 'NTCO3.SA', 'HAPV3.SA', 'MGLU3.SA', 'PRIO3.SA'
    ]

    metricas = []
    
    for ticker in pool_ativos:
        try:
            info = yf.Ticker(ticker).info
            metricas.append({
                'Papel': ticker.replace('.SA', ''),
                'DY': info.get('dividendYield', 0) or 0,
                'Crescimento': info.get('revenueGrowth', 0) or 0,
                'ROE': info.get('returnOnEquity', 0) or 0
            })
        except Exception:
            continue
            
    df = pd.DataFrame(metricas)
    
    # ESTRATÉGIA DE CONTINGÊNCIA (Fallback de Segurança)
    if df.empty:
        print("Aviso: Yahoo Finance não respondeu. Ativando Fallback de Segurança.")
        return ['WEGE3', 'ITUB4', 'VALE3'], "Eficiência/Blue Chips (Fallback)"
    
    # Filtro de Seleção Algorítmica com base na métrica chave de cada perfil
    if perfil == "Conservador":
        top3 = df.sort_values('DY', ascending=False).head(3)
        categoria_base = "Dividendos"
    elif perfil == "Audacioso":
        top3 = df.sort_values('Crescimento', ascending=False).head(3)
        categoria_base = "Crescimento Acelerado"
    else: # Moderado
        top3 = df.sort_values('ROE', ascending=False).head(3)
        categoria_base = "Eficiência/Blue Chips"
    
    tickers = top3['Papel'].tolist()
    return tickers, categoria_base

# ==============================================================================
# MOTOR GENERATIVO (Camada de Explicação Analítica Textual)
# ==============================================================================
def gerar_motivos_dinamicos(perfil, tickers):
    print("Acionando a API para redação autônoma das teses de investimento...")
    nomes_ativos = ", ".join(tickers)
    
    prompt = f"""
    Atue como um analista quantitativo.
    O algoritmo selecionou as ações: {nomes_ativos}. Perfil do cliente: {perfil}.
    Escreva um motivo de 1 linha para cada ação justificando a escolha.
    Responda APENAS em formato JSON válido, usando os tickers como chaves. Sem markdown.
    """
    try:
        resposta = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        texto_limpo = resposta.text.replace('```json', '').replace('```', '').strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro no Motor Generativo: {e}")
        return {t: "Ativo selecionado pelo filtro quantitativo avançado." for t in tickers}

# ==============================================================================
# ALOCAÇÃO DE CAPITAL E PRECIFICAÇÃO REALISTA (Cálculo de Cotas Inteiras)
# ==============================================================================
def calcular_aporte_dinamico(tickers_limpos, motivos, categoria_base, capital_disponivel):
    ativos_processados = []
    dinheiro_sobrando = capital_disponivel
    peso_por_ativo = 1.0 / len(tickers_limpos) 

    tickers_yf = [t + '.SA' for t in tickers_limpos]
    dados_historicos = yf.download(tickers_yf, period="1d", progress=False)['Close']

    for ticker_limpo in tickers_limpos:
        ticker_sa = ticker_limpo + '.SA'
        try:
            if ticker_sa in dados_historicos and not dados_historicos[ticker_sa].empty:
                preco_atual = float(dados_historicos[ticker_sa].iloc[-1])
            else:
                ticker_obj = yf.Ticker(ticker_sa)
                preco_atual = float(ticker_obj.history(period="1d")['Close'].iloc[-1])

            preco_atual = round(preco_atual, 2)
            fatia_dinheiro = capital_disponivel * peso_por_ativo
            
            quantidade = math.floor(fatia_dinheiro / preco_atual)
            total_investido = quantidade * preco_atual

            if quantidade > 0:
                ativos_processados.append({
                    "ativo": ticker_limpo,
                    "preco_atual": preco_atual,
                    "quantidade": quantidade,
                    "total_gasto": round(total_investido, 2),
                    "categoria": categoria_base.replace(" (Fallback)", ""),
                    "motivo": motivos.get(ticker_limpo, "Alinhado ao perfil estratégico.")
                })
                dinheiro_sobrando -= total_investido 
        except Exception as e:
            print(f"Erro ao precificar {ticker_limpo}: {e}")

    return ativos_processados, round(dinheiro_sobrando, 2)

# ==============================================================================
# MOTOR PSICOLÓGICO: REDE NEURAL RECORRENTE (PyTorch)
# ==============================================================================
class PerfilInvestidorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=3):
        super(PerfilInvestidorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout_fc(lstm_out[:, -1, :])
        return self.fc(out)

# ==============================================================================
# CARREGAMENTO DO MODELO E PARAMETRIZAÇÃO DE ESCALONAMENTO (Z-Score)
# ==============================================================================
try:
    checkpoint = torch.load('lstm_guide.pth', map_location=torch.device('cpu'), weights_only=False)
    modelo_ia = PerfilInvestidorLSTM(input_size=2, hidden_size=checkpoint['hidden_size'], num_layers=2)
    modelo_ia.load_state_dict(checkpoint['model_state_dict'])
    modelo_ia.eval() 
    
    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    mapeamento_reverso = {int(k): v for k, v in checkpoint['mapeamento_reverso'].items()}
    print("Pesos e parâmetros da Rede Neural carregados com sucesso!")
except Exception as e:
    print(f"Erro crítico ao inicializar o arquivo de pesos .pth: {e}")
    modelo_ia = None

# ==============================================================================
# ROTAS DA INTERFACE E ENDPOINTS DA API
# ==============================================================================
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/api/simular_ia', methods=['POST'])
def simular_ia():
    try:
        dados = request.json
        historico_bruto = dados.get('historico_10_anos')
        
        if not historico_bruto:
            return jsonify({"status": "erro", "mensagem": "Dados ausentes."}), 400
        
        hist_np = np.array([historico_bruto])
        hist_scaled = (hist_np - X_mean) / X_std
        tensor_entrada = torch.FloatTensor(hist_scaled)

        with torch.no_grad():
            saida_bruta = modelo_ia(tensor_entrada)
            
            temperatura = 2.5 
            probabilidades = F.softmax(saida_bruta / temperatura, dim=1)[0] * 100

        probabilidades_lista = probabilidades.tolist()
        indice_vencedor = torch.argmax(saida_bruta, dim=1).item()
        perfil_escolhido = mapeamento_reverso[indice_vencedor]

        return jsonify({
            "status": "sucesso",
            "perfil_vencedor": perfil_escolhido,
            "confianca_da_rede": {
                "Conservador": round(probabilidades_lista[0], 2),
                "Moderado": round(probabilidades_lista[1], 2),
                "Audacioso": round(probabilidades_lista[2], 2)
            }
        }), 200
    except Exception as e:
        print(f"Erro na inferência ao vivo: {e}")
        return jsonify({"status": "erro", "mensagem": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    try:
        dados = request.json
        email = dados.get('email')
        senha = dados.get('senha')

        with open('usuarios.json', 'r', encoding='utf-8') as f:
            banco = json.load(f)

        if email not in banco or banco[email]['senha'] != senha:
            return jsonify({"status": "erro", "mensagem": "Credenciais Inválidas."}), 401

        usuario = banco[email]
        perfil_ia = usuario['perfil_classificado_pela_ia']
        capital_livre = usuario['renda_atual'] * usuario['poupanca_atual']

        tickers_vencedores, categoria = buscar_melhores_acoes(perfil_ia)
        motivos_gerados = gerar_motivos_dinamicos(perfil_ia, tickers_vencedores)
        lista_de_compras, troco = calcular_aporte_dinamico(tickers_vencedores, motivos_gerados, categoria, capital_livre)

        return jsonify({
            "status": "sucesso",
            "dados_cliente": {
                "nome": usuario['nome'],
                "perfil_ia": perfil_ia,
                "historico": usuario.get('historico', [[0,0] for _ in range(120)]),
            },
            "guia_de_investimento": {
                "capital_investido": round(capital_livre - troco, 2),
                "capital_sobrando": troco,
                "alocacao": lista_de_compras
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 400

@app.route('/api/alterar_perfil_alocacao', methods=['POST'])
def alterar_perfil_alocacao():
    try:
        dados = request.json
        email = dados.get('email')
        novo_perfil = dados.get('perfil') 

        with open('usuarios.json', 'r', encoding='utf-8') as f:
            banco = json.load(f)

        usuario = banco[email]
        capital_livre = usuario['renda_atual'] * usuario['poupanca_atual']

        tickers_vencedores, categoria = buscar_melhores_acoes(novo_perfil)
        motivos_gerados = gerar_motivos_dinamicos(novo_perfil, tickers_vencedores)
        lista_de_compras, troco = calcular_aporte_dinamico(tickers_vencedores, motivos_gerados, categoria, capital_livre)

        return jsonify({
            "status": "sucesso",
            "guia_de_investimento": {
                "capital_investido": round(capital_livre - troco, 2),
                "capital_sobrando": troco,
                "alocacao": lista_de_compras
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)