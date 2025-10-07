# app.py (FINAL WORKING VERSION)

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. DEFINIÇÃO DO PROMPT DO AGENTE FISCAL ---
PREFIXO_AGENTE_FISCAL = """
Você é um assistente fiscal de IA, especialista em tributação para PMEs no Brasil. Sua função é analisar um dataframe de notas fiscais.

### REGRAS:
1.  **FOCO TOTAL NOS DADOS:** Responda APENAS com base no dataframe `df`. Se a pergunta for sobre outro assunto, recuse educadamente.
2.  **CÁLCULO DE IMPOSTOS:** Para calcular impostos, primeiro use o dataframe para encontrar a receita. Depois, se não souber a alíquota, PEÇA ao usuário por ela. Não invente valores.
3.  **RESPOSTA ESTRUTURADA:** Apresente o resultado direto e depois uma breve explicação de como chegou a ele.
"""

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="Agente Fiscal Inteligente", page_icon="🧾", layout="wide")
st.title("🧾 Agente Fiscal Inteligente")
st.write("Um assistente de IA para análise e pré-apuração de impostos.")

# --- Inicialização do Estado da Sessão ---
if "google_api_key" not in st.session_state: st.session_state.google_api_key = None
if "df" not in st.session_state: st.session_state.df = None
if "agent" not in st.session_state: st.session_state.agent = None
if "messages" not in st.session_state: st.session_state.messages = []
if "uploaded_file_name" not in st.session_state: st.session_state.uploaded_file_name = None

# --- Barra Lateral (Sidebar) para Configurações ---
with st.sidebar:
    st.header("1. Configuração da API")
    api_key_input = st.text_input("Chave da API do Google", type="password")
    if api_key_input:
        st.session_state.google_api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
    if st.session_state.google_api_key: st.sidebar.success("API do Google configurada.", icon="🔑")

    st.header("2. Perfil da Empresa")
    regime_tributario = st.selectbox("Regime Tributário", ["Simples Nacional", "Lucro Presumido"])
    faturamento_anual = st.number_input("Faturamento Anual Acumulado (R$)", min_value=0.0, step=1000.0)
    cnae = st.text_input("Atividade Principal (CNAE)", placeholder="Ex: 4781-4/00")

    st.header("3. Upload dos Documentos")
    arquivo = st.file_uploader("Faça o upload de um arquivo (CSV ou Excel)", type=["csv", "xlsx"])
    if arquivo is not None and st.session_state.get('uploaded_file_name') != arquivo.name:
        try:
            st.session_state.df = pd.read_csv(arquivo) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo)
            st.session_state.uploaded_file_name = arquivo.name
            st.success(f"Arquivo '{arquivo.name}' carregado!", icon="📄")
            st.session_state.agent = None
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            st.session_state.df = None

# --- Lógica Principal da Aplicação ---
if st.session_state.google_api_key and st.session_state.df is not None:
    
    st.header("Dashboard Gerencial")
    try:
        df_vendas = st.session_state.df[st.session_state.df['NATUREZA DA OPERAÇÃO'].str.contains("VENDA", case=False)]
        faturamento_total = df_vendas['VALOR TOTAL'].sum()
        num_notas = len(df_vendas['NÚMERO'].unique())
        ticket_medio = faturamento_total / num_notas if num_notas > 0 else 0
        num_clientes = df_vendas['NOME DESTINATÁRIO'].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faturamento Bruto (Vendas)", f"R$ {faturamento_total:,.2f}")
        col2.metric("Notas Fiscais de Venda", num_notas)
        col3.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
        col4.metric("Clientes Únicos", num_clientes)
    except Exception:
        st.warning("Não foi possível gerar o dashboard. Verifique as colunas do seu arquivo.")

    st.header("Chat com o Agente Fiscal")

    if st.session_state.agent is None:
        st.info("Inicializando o agente fiscal...")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_version="v1")
            
            prompt_formatado = PREFIXO_AGENTE_FISCAL.format(
                regime_tributario=regime_tributario,
                faturamento_anual=faturamento_anual,
                cnae=cnae
            )

            st.session_state.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=st.session_state.df,
                agent_type='tool-calling',
                prefix=prompt_formatado,
                verbose=True,
                handle_parsing_errors=True
            )
            st.success("Agente pronto para analisar seus dados!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou seu assistente fiscal. Analise o dashboard e me faça uma pergunta."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Qual o valor total das vendas este mês?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O agente está analisando..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    output_text = response["output"]
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Por favor, preencha a API Key e carregue um arquivo para começar.")
