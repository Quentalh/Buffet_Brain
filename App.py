from flask import Flask, request, jsonify, render_template
import yfinance as yf
import math
import json

app = Flask(__name__)

def calcular_aporte_dinamico(ativos_vencedores, capital_disponivel):
    ativos_processados = []
    dinheiro_sobrando = capital_disponivel
    peso_por_ativo = 1.0 / len(ativos_vencedores) 
    
    for ativo in ativos_vencedores:
        preco_atual = ativo['preco_atual']
        fatia_dinheiro = capital_disponivel * peso_por_ativo
        quantidade = math.floor(fatia_dinheiro / preco_atual)
        total_investido = quantidade * preco_atual
        
        if quantidade > 0:
            ativos_processados.append({
                "ativo": ativo['simbolo'].replace('.SA', ''),
                "nome": ativo['nome'],
                "preco_atual": preco_atual,
                "quantidade": quantidade,
                "total_gasto": round(total_investido, 2),
                "categoria": ativo['categoria'],
                "motivo": f"{ativo['motivo_base']} (Métrica: {ativo['metrica_calculada']:.2f})"
            })
            dinheiro_sobrando -= total_investido
            
    return ativos_processados, round(dinheiro_sobrando, 2)

def gerar_carteira(perfil, renda, poupanca):
    capital_livre = renda * poupanca
    
    pool_mercado = {
        "Conservador": [
            {"simbolo": "B5P211.SA", "nome": "ETF Tesouro", "categoria": "Renda Fixa", "motivo_base": "Proteção base."},
            {"simbolo": "KNCR11.SA", "nome": "Kinea Rend", "categoria": "FII", "motivo_base": "Atrelado à Selic."},
            {"simbolo": "TAEE11.SA", "nome": "Taesa S.A.", "categoria": "Ações Perenes", "motivo_base": "Transmissão elétrica."}
        ],
        "Moderado": [
            {"simbolo": "ITUB4.SA", "nome": "Itaú Unibanco", "categoria": "Setor Financeiro", "motivo_base": "Solidez corporativa."},
            {"simbolo": "HGLG11.SA", "nome": "CSHG Log", "categoria": "Fundo Imobiliário", "motivo_base": "Galpões logísticos."},
            {"simbolo": "IVVB11.SA", "nome": "ETF S&P 500", "categoria": "Global", "motivo_base": "Exposição aos EUA."}
        ],
        "Audacioso": [
            {"simbolo": "WEGE3.SA", "nome": "WEG S.A.", "categoria": "Indústria", "motivo_base": "Crescimento acelerado."},
            {"simbolo": "PETR4.SA", "nome": "Petrobras", "categoria": "Commodities", "motivo_base": "Alto dividend yield."},
            {"simbolo": "HASH11.SA", "nome": "ETF Cripto", "categoria": "Criptomoedas", "motivo_base": "Volatilidade alta."}
        ]
    }
    
    ativos_perfil = {item['simbolo']: item for item in pool_mercado[perfil]}
    lista_tickers = list(ativos_perfil.keys())
    
    dados_historicos = yf.download(lista_tickers, period="1mo", progress=False)['Close']
    ativos_ranqueados = []
    
    for ticker in lista_tickers:
        try:
            precos = dados_historicos[ticker].dropna()
            if len(precos) >= 2:
                preco_atual = float(precos.iloc[-1])
                retorno_mensal = ((preco_atual / float(precos.iloc[0])) - 1) * 100
                
                ativo_info = ativos_perfil[ticker].copy()
                ativo_info.update({"preco_atual": preco_atual, "metrica_calculada": retorno_mensal})
                ativos_ranqueados.append(ativo_info)
        except Exception:
            continue
            
    selecionados = sorted(ativos_ranqueados, key=lambda x: x['metrica_calculada'], reverse=True)[:3]
    
    lista_de_compras, troco = calcular_aporte_dinamico(selecionados, capital_livre)
    
    return {
        "capital_investido": round(capital_livre - troco, 2),
        "capital_sobrando": troco,
        "alocacao": lista_de_compras
    }



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
        
        guia_investimento = gerar_carteira(perfil_ia, usuario['renda_atual'], usuario['poupanca_atual'])
        
        return jsonify({
            "status": "sucesso",
            "dados_cliente": {
                "nome": usuario['nome'],
                "perfil_ia": perfil_ia
            },
            "guia_de_investimento": guia_investimento
        }), 200
        
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)