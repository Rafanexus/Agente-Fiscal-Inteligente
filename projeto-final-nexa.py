import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
# Importações para o novo agente com múltiplas ferramentas
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import PythonAstREPLTool

# --- 1. DEFINIÇÃO DO PROMPT DO AGENTE FISCAL ---
PREFIXO_AGENTE_FISCAL_OTIMIZADO = """
Você é um assistente fiscal de IA para PMEs no Brasil. Sua função é responder perguntas usando as ferramentas disponíveis.

### FERRAMENTAS DISPONÍVEIS ###
1.  **pandas_tool**: Use esta ferramenta para analisar o dataframe `df` de notas fiscais carregado pelo usuário. Ela pode responder perguntas sobre faturamento, produtos, clientes, etc.
2.  **web_search**: Use esta ferramenta para buscar informações atualizadas na internet que não estão no dataframe, como alíquotas de impostos, regras fiscais, legislação e tabelas do Simples Nacional.

### REGRAS PRINCIPAIS ###
1.  **Estrutura da Resposta Final:** Apresente primeiro o resultado direto e, em seguida, uma breve explicação de como você chegou a ele. Use formatação de markdown para clareza.
2.  **FOCO TOTAL NOS DADOS E BUSCA:** Responda APENAS com base no dataframe `df` ou em informações obtidas pela ferramenta `web_search`. Se a pergunta for sobre outro assunto, recuse educadamente.
3.  **Pense Passo a Passo:** Para perguntas complexas, explique seu plano. Ex: "Para calcular o imposto, primeiro usarei a ferramenta `pandas_tool` para obter o faturamento do mês. Depois, usarei a `web_search` para encontrar a alíquota correta. Por fim, farei o cálculo."
"""

# --- Configuração da Página Streamlit ---
st.set_page_config(page_title="Agente Fiscal Inteligente", page_icon="🧾", layout="wide")
st.title("🧾 Agente Fiscal Inteligente")
st.write("Análise fiscal, cálculo de impostos e insights com o poder da IA e busca na web.")

# --- Inicialização do Estado da Sessão ---
# ... (sem mudanças aqui) ...
if "google_api_key" not in st.session_state: st.session_state.google_api_key = None
if "tavily_api_key" not in st.session_state: st.session_state.tavily_api_key = None # Nova chave
if "df" not in st.session_state: st.session_state.df = None
if "agent" not in st.session_state: st.session_state.agent = None
if "messages" not in st.session_state: st.session_state.messages = []
if "uploaded_file_name" not in st.session_state: st.session_state.uploaded_file_name = None

# --- Barra Lateral (Sidebar) para Configurações ---
with st.sidebar:
    st.header("1. Configuração das APIs")
    
    # Chave do Google
    google_api_key_input = st.text_input("Chave da API do Google", type="password", help="Necessária para o modelo de linguagem.")
    if google_api_key_input:
        st.session_state.google_api_key = google_api_key_input
        os.environ["GOOGLE_API_KEY"] = google_api_key_input

    # Chave da Tavily
    tavily_api_key_input = st.text_input("Chave da API da Tavily", type="password", help="Necessária para a busca na web.")
    if tavily_api_key_input:
        st.session_state.tavily_api_key = tavily_api_key_input
        os.environ["TAVILY_API_KEY"] = tavily_api_key_input

    # Confirmação visual
    if st.session_state.google_api_key: st.sidebar.success("API do Google configurada.", icon="🔑")
    if st.session_state.tavily_api_key: st.sidebar.success("API da Tavily configurada.", icon="🔎")

    st.header("2. Perfil da Empresa")
    # ... (sem mudanças aqui) ...
    regime_tributario = st.selectbox("Regime Tributário", ["Simples Nacional", "Lucro Presumido"])
    faturamento_anual = st.number_input("Faturamento Anual Acumulado (R$)", min_value=0.0, step=1000.0)
    cnae = st.text_input("Atividade Principal (CNAE)", placeholder="Ex: 4781-4/00")

    st.header("3. Upload dos Documentos")
    # ... (sem mudanças aqui) ...
    arquivo = st.file_uploader("Faça o upload de um arquivo (CSV ou Excel)", type=["csv", "xlsx"])
    if arquivo is not None and st.session_state.get('uploaded_file_name') != arquivo.name:
        try:
            if arquivo.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(arquivo)
            else:
                st.session_state.df = pd.read_excel(arquivo)
            st.session_state.uploaded_file_name = arquivo.name
            st.success(f"Arquivo '{arquivo.name}' carregado!", icon="📄")
            st.session_state.agent = None
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")
            st.session_state.df = None

# --- Lógica Principal da Aplicação ---
if st.session_state.google_api_key and st.session_state.tavily_api_key and st.session_state.df is not None:
    
    st.header("Dashboard Gerencial")
    # --- NOVO DASHBOARD MELHORADO ---
    try:
        df_vendas = st.session_state.df[st.session_state.df['NATUREZA DA OPERAÇÃO'].str.contains("VENDA", case=False)]
        
        # KPIs
        faturamento_total = df_vendas['VALOR TOTAL'].sum()
        num_notas = len(df_vendas['NÚMERO'].unique())
        ticket_medio = faturamento_total / num_notas if num_notas > 0 else 0
        num_clientes = df_vendas['NOME DESTINATÁRIO'].nunique()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Faturamento Bruto (Vendas)", f"R$ {faturamento_total:,.2f}")
        col2.metric("Notas Fiscais de Venda", num_notas)
        col3.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
        col4.metric("Clientes Únicos", num_clientes)

        st.divider()

        # Gráficos
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 5 Clientes por Faturamento")
            top_5_clientes = df_vendas.groupby('NOME DESTINATÁRIO')['VALOR TOTAL'].sum().nlargest(5)
            st.bar_chart(top_5_clientes)
        with col2:
            st.subheader("Vendas por Estado (UF)")
            vendas_por_uf = df_vendas.groupby('UF DESTINATÁRIO')['VALOR TOTAL'].sum()
            st.bar_chart(vendas_por_uf)

    except Exception as e:
        st.warning(f"Não foi possível gerar o dashboard completo. Verifique as colunas do seu arquivo. Erro: {e}")

    st.header("Chat com o Agente Fiscal")

# --- NOVA ARQUITETURA DO AGENTE (VERSÃO FINAL E MAIS ESTÁVEL) ---
if st.session_state.agent is None:
    st.info("Inicializando o agente fiscal com busca na web...")
    try:
        # Importações necessárias DENTRO do bloco, para clareza
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_community.tools import PythonAstREPLTool # A NOVA FERRAMENTA

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_version="v1")
        
        # 1. CRIAR A FERRAMENTA DE ANÁLISE DE DADOS MANUALMENTE
        # Damos à ferramenta acesso ao nosso dataframe 'df'
        pandas_tool = PythonAstREPLTool(
            name="analise_documento_fiscal",
            description="Use esta ferramenta para fazer qualquer análise, cálculo ou pergunta sobre o dataframe `df` que contém os dados das notas fiscais. A ferramenta executa código Python. O input para a ferramenta deve ser um código Python válido para responder à pergunta do usuário.",
            locals={"df": st.session_state.df} # AQUI ESTÁ A MÁGICA: damos acesso ao 'df'
        )

        # 2. Criar a Ferramenta de Busca na Web (como antes)
        search_tool = TavilySearchResults(max_results=3)
        search_tool.name = "busca_web_para_informacoes_fiscais"
        search_tool.description = "Use esta ferramenta para buscar informações atualizadas na internet que não estão no documento, como alíquotas de impostos, regras fiscais, legislação, tabelas do Simples Nacional e significado de códigos como CFOP."
        
        # 3. Juntar as ferramentas em uma lista
        tools = [pandas_tool, search_tool]
        
        # 4. Prompt do Agente (ajustado para a nova ferramenta)
        # Não precisamos mais do prefixo, pois o prompt do agente é mais explícito
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente fiscal de IA para PMEs no Brasil. Sua função é responder perguntas usando as ferramentas disponíveis.
            
            ### FERRAMENTAS:
            1. `analise_documento_fiscal`: Executa código Python para analisar o dataframe `df` com os dados fiscais. Use-a para calcular faturamento, filtrar notas, etc.
            2. `busca_web_para_informacoes_fiscais`: Busca informações atualizadas na web, como alíquotas e leis.

            ### FLUXO DE TRABALHO:
            - Para perguntas sobre os dados do arquivo, use a ferramenta `analise_documento_fiscal`.
            - Para perguntas sobre leis, alíquotas ou informações externas, use `busca_web_para_informacoes_fiscais`.
            - Para perguntas complexas (ex: calcular imposto), combine as ferramentas: primeiro use `analise_documento_fiscal` para obter o faturamento, depois use `busca_web_para_informacoes_fiscais` para obter a alíquota, e então faça o cálculo final.
            - Responda de forma clara e estruturada.
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # 5. Criação do Agente e do Executor
        agent = create_tool_calling_agent(llm, tools, prompt_template)
        st.session_state.agent = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )
        
        st.success("Agente com acesso à internet pronto!")
    except Exception as e:
        st.error(f"Erro ao criar o agente: {e}")
        st.stop()

    # Lógica do Chat (com o prompt aumentado)
    if not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": "Olá! Sou seu assistente fiscal. Analise o dashboard e me faça uma pergunta."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Qual a alíquota do Simples Nacional para meu faturamento?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando e pesquisando..."):
                try:
                    prompt_aumentado = f"""
                    CONTEXTO DA EMPRESA:
                    - Regime: {regime_tributario}
                    - Faturamento Anual: R$ {faturamento_anual}
                    - CNAE: {cnae}
                    
                    PERGUNTA: {prompt}
                    """
                    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    response = st.session_state.agent.invoke({"input": prompt_aumentado, "chat_history": chat_history})
                    output_text = response["output"]
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Por favor, preencha as configurações de API e empresa, e faça o upload de um arquivo para começar.")
