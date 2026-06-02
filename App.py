from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
from google import genai
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


app = Flask(__name__)

client = genai.Client(api_key=" Chave placeholder | Para o programa rodar precisa ser inserida")

def buscar_melhores_acoes(perfil):
    print("Iniciando varredura quantitativa via API Global (Yahoo Finance)...")

    pool_ativos = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBAS3.SA', 'WEGE3.SA',
        'TAEE11.SA', 'PRIO3.SA', 'EGIE3.SA', 'BBDC4.SA', 'VIVT3.SA',
        'SUZB3.SA', 'TOTS3.SA', 'RENT3.SA', 'EQTL3.SA', 'RADL3.SA'
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

    if perfil == "Conservador":
        top3 = df.sort_values('DY', ascending=False).head(3)
        categoria_base = "Dividendos"
    elif perfil == "Audacioso":
        top3 = df.sort_values('Crescimento', ascending=False).head(3)
        categoria_base = "Crescimento Acelerado"
    else:
        top3 = df.sort_values('ROE', ascending=False).head(3)
        categoria_base = "Eficiência/Blue Chips"

    tickers = top3['Papel'].tolist() if not df.empty else ['WEGE3', 'ITUB4', 'VALE3']
    return tickers, categoria_base

def gerar_motivos_dinamicos(perfil, tickers):
    print("Acionando o Gemini para redação dos relatórios...")
    nomes_ativos = ", ".join(tickers)

    prompt = f"""
    Atue como um analista quantitativo.
    O algoritmo matemático selecionou as seguintes ações brasileiras para investir hoje: {nomes_ativos}.
    O cliente tem o perfil {perfil}.

    Escreva um motivo de exata 1 linha para cada ação, justificando financeiramente por que ela se encaixa neste perfil.
    Responda APENAS no formato JSON, usando os tickers como chaves, sem markdown.
    Exemplo:
    {{
        "{tickers[0]}": "motivo aqui",
        "{tickers[1]}": "motivo aqui"
    }}
    """
    try:
            resposta = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            texto_limpo = resposta.text.replace('```json', '').replace('```', '').strip()
            motivos = json.loads(texto_limpo)
            return motivos
    except Exception as e:
            print(f"Erro no Gemini: {e}")
            return {t: "Ativo selecionado pelo filtro quantitativo avançado." for t in tickers}


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
                categoria_limpa = categoria_base.replace(" (Fallback)", "")

                link_verificacao = f"https://statusinvest.com.br/acoes/{ticker_limpo.lower()}"

                ativos_processados.append({
                    "ativo": ticker_limpo,
                    "nome": ticker_limpo,
                    "preco_atual": preco_atual,
                    "quantidade": quantidade,
                    "total_gasto": round(total_investido, 2),
                    "categoria": categoria_limpa,
                    "motivo": motivos.get(ticker_limpo, "Ativo com forte alinhamento ao seu perfil estratégico."),
                    "link": link_verificacao
                })
                dinheiro_sobrando -= total_investido
        except Exception as e:
            print(f"Erro ao precificar {ticker_limpo}: {e}")

    return ativos_processados, round(dinheiro_sobrando, 2)


class PerfilInvestidorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=16, num_layers=1, num_classes=3):
        super(PerfilInvestidorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


try:
    modelo_ia = PerfilInvestidorLSTM()
    modelo_ia.load_state_dict(torch.load('pesos_modelo.pth'))
    modelo_ia.eval()
    print("Rede Neural carregada com sucesso!")
except FileNotFoundError:
    print("Aviso: pesos_modelo.pth não encontrado. Usando pesos aleatórios para simulação da interface.")
    modelo_ia = PerfilInvestidorLSTM()
    modelo_ia.eval()


@app.route('/api/simular_ia', methods=['POST'])
def simular_ia():
    try:
        dados = request.json
        historico = dados.get('historico_4_anos')
        tensor_entrada = torch.tensor([historico], dtype=torch.float32)

        with torch.no_grad():
            saida_bruta = modelo_ia(tensor_entrada)
            probabilidades = F.softmax(saida_bruta, dim=1)[0] * 100

        probabilidades_lista = probabilidades.tolist()

        perfis = ["Conservador", "Moderado", "Audacioso"]
        indice_vencedor = torch.argmax(saida_bruta, dim=1).item()
        perfil_escolhido = perfis[indice_vencedor]

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
        return jsonify({"status": "erro", "mensagem": str(e)}), 400


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/api/login', methods=['POST'])
def login():
    try:
        dados = request.json
        email = dados.get('email')
        senha = dados.get('senha')

        with open('usuarios.json', 'r', encoding='utf-8') as f:
            banco = json.load(f)

        if email not in banco or banco[email]['senha'] != senha:
            return jsonify({"status": "erro", "mensagem": "Email ou senha incorretos."}), 401

        usuario = banco[email]
        perfil_ia = usuario['perfil_classificado_pela_ia']
        capital_livre = usuario['renda_atual'] * usuario['poupanca_atual']

        tickers_vencedores, categoria = buscar_melhores_acoes(perfil_ia)
        motivos_gerados = gerar_motivos_dinamicos(perfil_ia, tickers_vencedores)
        lista_de_compras, troco = calcular_aporte_dinamico(tickers_vencedores, motivos_gerados, categoria, capital_livre)

        guia_investimento = {
            "capital_investido": round(capital_livre - troco, 2),
            "capital_sobrando": troco,
            "alocacao": lista_de_compras
        }

        return jsonify({
            "status": "sucesso",
            "dados_cliente": {
                "nome": usuario['nome'],
                "perfil_ia": perfil_ia,
                "historico": usuario.get('historico', [[0,0],[0,0],[0,0],[0,0]])
            },
            "guia_de_investimento": guia_investimento
        }), 200

    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
