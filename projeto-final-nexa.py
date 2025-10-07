import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. DEFINIÇÃO DO PROMPT DO AGENTE FISCAL ---
# Este é o "cérebro" do nosso agente. Ele será formatado com os dados da empresa.
PREFIXO_AGENTE_FISCAL = """
Você é um assistente fiscal de IA, especialista em tributação para pequenas empresas no Brasil. Sua função é analisar dados de notas fiscais (contidos em um dataframe) e responder a perguntas com base no perfil da empresa fornecido.

### PERFIL DA EMPRESA PARA ANÁLISE ###
- **Regime Tributário:** {regime_tributario}
- **Faturamento Anual Acumulado (últimos 12 meses):** R$ {faturamento_anual}
- **Atividade Principal (CNAE):** {cnae}

### REGRAS TRIBUTÁRIAS BÁSICAS (SIMPLIFICADAS PARA ESTE MVP) ###
- **Se o Regime for 'Simples Nacional'**:
  1. O imposto principal é o DAS (Documento de Arrecadação do Simples Nacional).
  2. Para calcular o DAS, você deve primeiro somar o valor de todas as notas fiscais de 'Venda' (saída) no período analisado.
  3. Use a receita bruta do mês para encontrar a alíquota. Para este MVP, pode usar uma tabela simplificada ou pedir ao usuário a alíquota se não souber. Ex: "Com base em um faturamento de X, a alíquota do Simples é Y%".
  4. O valor do DAS é (Receita Bruta do Mês * Alíquota).
  5. Sempre explique como chegou ao cálculo.

- **Se o Regime for 'Lucro Presumido'**:
  1. Os impostos são calculados separadamente (PIS, COFINS, IRPJ, CSLL).
  2. A base de cálculo do IRPJ é 8% da receita para comércio e 32% para serviços. A alíquota do imposto é 15%.
  3. PIS é 0.65% e COFINS é 3% sobre o faturamento total.
  4. Explique cada cálculo separadamente.

### SUAS TAREFAS E COMPORTAMENTO ###
1.  **Estrutura da Resposta Final:** Ao formular a resposta para o usuário, apresente primeiro o resultado direto e, em seguida, uma breve explicação de como você chegou a ele. Mantenha a explicação clara e use formatação de markdown (negrito, listas) para facilitar a leitura.  
2.  **Foco Exclusivo Fiscal:** Responda APENAS a perguntas relacionadas à análise fiscal do documento. Recuse educadamente qualquer pergunta fora do tópico (ex: "qual a capital da França?").
3.  **Use o Perfil da Empresa:** Sempre leve em conta o regime tributário e o faturamento informados para seus cálculos e respostas.
4.  **Pense Passo a Passo:** Explique seu raciocínio antes de executar um cálculo.
5.  **Verificação Inicial:** Ao analisar um novo arquivo, comece com `df.info()` e `df.head()` para entender os dados.
"""

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Agente Fiscal Inteligente",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Agente Fiscal Inteligente (MVP v1)")
st.write("Um assistente de IA para análise e pré-apuração de impostos a partir de seus documentos fiscais.")

# --- Inicialização do Estado da Sessão (Session State) ---
# Usado para armazenar dados entre as interações do usuário
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = None
if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# --- Barra Lateral (Sidebar) para Configurações ---
with st.sidebar:
    st.header("1. Configuração Inicial")
    
    # Campo para a API Key do Google, sempre visível
    api_key_input = st.text_input(
        "Chave da API do Google", 
        type="password", 
        help="Insira sua chave da API do Google aqui."
    )

    # Lógica Híbrida para carregar a chave
    # Tenta carregar dos segredos primeiro (para o seu uso no deploy)
    try:
        if "GOOGLE_API_KEY" in st.secrets and st.secrets["GOOGLE_API_KEY"]:
            st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
            # Mensagem discreta para confirmar que o segredo foi usado
            st.sidebar.success("Chave da API carregada dos segredos.", icon="✅")
    except:
        # Se st.secrets não existir ou falhar, não faz nada, dependerá do input abaixo
        pass

    # Se a chave ainda não foi carregada pelos segredos, usa o input do usuário
    if not st.session_state.get("google_api_key") and api_key_input:
        st.session_state.google_api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.sidebar.success("Chave da API configurada.", icon="🔑")

    # O resto da sua sidebar continua aqui...
    st.header("2. Perfil da Empresa")
    regime_tributario = st.selectbox(
        "Regime Tributário",
        ["Simples Nacional", "Lucro Presumido", "Lucro Real (Não implementado)"]
    )
    faturamento_anual = st.number_input("Faturamento Anual Acumulado (R$)", min_value=0.0, step=1000.0)
    cnae = st.text_input("Atividade Principal (CNAE)", placeholder="Ex: 4781-4/00 (Comércio)")

    st.header("3. Upload dos Documentos")
    arquivo = st.file_uploader(
        "Faça o upload de um arquivo (CSV ou Excel) com os dados das NFs",
        type=["csv", "xlsx"]
    )

    # Lógica para processar o arquivo apenas uma vez
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

# Verifica se todas as condições para iniciar foram atendidas
if st.session_state.google_api_key and st.session_state.df is not None:
    
    st.header("Dashboard Gerencial")
    try:
        # Lista de colunas necessárias para o dashboard
        colunas_necessarias = ['NATUREZA DA OPERAÇÃO', 'VALOR TOTAL', 'NÚMERO', 'NOME DESTINATÁRIO']
    
        # Verifica se todas as colunas existem no dataframe
        if all(coluna in st.session_state.df.columns for coluna in colunas_necessarias):
            df_vendas = st.session_state.df[st.session_state.df['NATUREZA DA OPERAÇÃO'].str.contains("VENDA", case=False, na=False)]
        
            faturamento_total = df_vendas['VALOR TOTAL'].sum()
            num_notas = len(df_vendas['NÚMERO'].unique())
            ticket_medio = faturamento_total / num_notas if num_notas > 0 else 0
            num_clientes = df_vendas['NOME DESTINATÁRIO'].nunique()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faturamento Bruto (Vendas)", f"R$ {faturamento_total:,.2f}")
            col2.metric("Notas Fiscais de Venda", num_notas)
            col3.metric("Ticket Médio", f"R$ {ticket_medio:,.2f}")
            col4.metric("Clientes Únicos", num_clientes)
        else:
            # Mensagem mais específica se as colunas estiverem faltando
            st.warning("Dashboard não pôde ser gerado. Verifique se o arquivo carregado contém as colunas necessárias: 'NATUREZA DA OPERAÇÃO', 'VALOR TOTAL', etc.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao gerar o dashboard: {e}")

    st.header("Chat com o Agente Fiscal")

    # Inicializa o agente se ele ainda não existir nesta sessão
    if st.session_state.agent is None:
        st.info("Inicializando o agente fiscal...")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                convert_system_message_to_human=True,
                api_version="v1"
            )
            
            # Formata o prompt com os dados da empresa
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
                handle_parsing_errors=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                allow_dangerous_code=True
            )
            st.success("Agente pronto para analisar seus dados!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    # Lógica do Chat (igual ao projeto anterior)
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Olá! Sou seu assistente fiscal. O que você gostaria de saber sobre os documentos carregados?"
        })

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
                    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    response = st.session_state.agent.invoke({"input": prompt, "chat_history": chat_history})
                    output_text = response["output"]
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Por favor, preencha todas as configurações na barra lateral para começar.")

