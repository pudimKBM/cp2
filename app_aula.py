#!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os # Ensure this is imported
from joblib import Memory # Ensure this is imported
import plotly.express as px 
from pycaret.classification import *
import openai
import plotly.graph_objects as go # Added for gauge chart
import re


st.set_page_config( page_title = 'Simulador - Case Ifood',
                    page_icon = './images/logo_fiap.png',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

# Initialize session state for treshold if it doesn't exist
if 'treshold' not in st.session_state:
    st.session_state.treshold = 0.50 # Default value with two decimal places

# Trigger para atualização instantânea do threshold ao clicar em atalhos
if 'treshold_update_trigger' not in st.session_state:
    st.session_state.treshold_update_trigger = 0

# Initialize OpenAI API key (in a real application, you would use st.secrets for this)
# This is a placeholder - the user will need to provide their own API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    
# Para rastrear qual foi o último atalho selecionado (para mostrar o check)
if 'ultimo_atalho' not in st.session_state:
    st.session_state.ultimo_atalho = ""

# Callbacks for treshold controls
def sync_treshold_from_slider():
    # Slider widget key is 'slider_widget_for_treshold'
    st.session_state.treshold = round(st.session_state.slider_widget_for_treshold, 2)

def sync_treshold_from_text():
    # Text input widget key is 'text_widget_for_treshold'
    try:
        val = float(st.session_state.text_widget_for_treshold)
        if 0.0 <= val <= 1.0:
            st.session_state.treshold = round(val, 2)
        else:
            # If invalid, the text input will revert to displaying the current st.session_state.treshold on the next rerun
            st.warning("Treshold via texto deve estar entre 0.0 e 1.0. Valor não aplicado.")
    except ValueError:
        # If invalid, the text input will revert to displaying the current st.session_state.treshold on the next rerun
        st.warning("Valor inválido para treshold. Por favor, insira um número. Valor não aplicado.")


        
def process_natural_language_treshold(prompt_text=None):
    # Permite passar o texto diretamente (atalho) ou usar o session_state
    if prompt_text is not None:
        user_prompt = prompt_text.strip()
    else:
        user_prompt = st.session_state.ai_prompt_input.strip() if 'ai_prompt_input' in st.session_state else ''
    if not user_prompt:
        return
    if not st.session_state.openai_api_key:
        st.warning("Por favor, insira uma chave de API OpenAI válida nas configurações.")
        return
    try:
        openai.api_key = st.session_state.openai_api_key
        messages = [
            {"role": "system", "content": """
            Você é um assistente especializado em machine learning que ajuda a definir thresholds para modelos de classificação binária.
            
            Regras importantes:
            1. Se o usuário mencionar um valor numérico específico (ex: "coloque o threshold em 0.8"), use EXATAMENTE esse valor.
            2. Se o usuário mencionar um valor de precisão (ex: "aumentar a precisão para 0.8"), use EXATAMENTE 0.8 como threshold.
            3. Se o usuário pedir para aumentar o threshold sem especificar um valor, adicione 0.1 ao valor atual.
            4. Se o usuário pedir para diminuir o threshold sem especificar um valor, subtraia 0.1 do valor atual.
            5. Se o usuário mencionar "modelo conservador", use um threshold mais alto (0.7 ou mais).
            6. Se o usuário mencionar "priorizar recall", use um threshold mais baixo (0.3 ou menos).
            7. Se o usuário mencionar "priorizar precisão" ou "priorizar precision", use um threshold bem alto (0.8 ou mais).
            8. Se o usuário mencionar "priorizar F1-score" ou "maximizar F1-score", utilize o valor do threshold que maximiza o F1-score (normalmente próximo de 0.5).
            9. Se o usuário mencionar "maximizar lucro", "minimizar custos" ou "threshold ótimo", utilize o valor ótimo para maximizar o retorno financeiro, normalmente entre 0.4 e 0.6, a menos que o usuário especifique custos.
            10. Se o usuário mencionar "priorizar especificidade" ou "especificidade", use um threshold alto (0.8 ou mais).
            11. Se o usuário mencionar "priorizar sensibilidade" ou "sensibilidade", use um threshold baixo (0.2 ou menos).
            12. Se o usuário mencionar "threshold ótimo pela curva ROC", utilize o valor do threshold que maximiza a soma de sensibilidade e especificidade (normalmente próximo de 0.5).
            
            Responda APENAS com um número entre 0.0 e 1.0, sem texto adicional.
            """},
            {"role": "user", "content": f"O threshold atual do meu modelo é {st.session_state.treshold}. O modelo prevê se um cliente comprará um produto (1) ou não (0). Solicitação: {user_prompt}. Com base nessa solicitação, qual seria um valor de threshold apropriado entre 0.0 e 1.0?"}
        ]
        try:
            client = openai.OpenAI(api_key=st.session_state.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=50,
                temperature=0.3,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            ai_response = response.choices[0].message.content.strip()
        except AttributeError:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=50,
                temperature=0.3,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            ai_response = response.choices[0].message.content.strip()
        match = re.search(r'([0-9]*\.?[0-9]+)', ai_response)
        if match:
            suggested_threshold = float(match.group(1))
            if 0.0 <= suggested_threshold <= 1.0:
                st.session_state.treshold = round(suggested_threshold, 2)
                st.session_state.treshold_update_trigger += 1
                # Forçar rerun imediato para atualizar todos os componentes
                # st.rerun()
                # O código abaixo só será executado após o rerun
                st.success(f"Threshold ajustado para {st.session_state.treshold} com base na sua solicitação.")
            else:
                st.warning(f"O valor sugerido ({suggested_threshold}) está fora do intervalo válido (0.0-1.0). Nenhuma alteração foi feita.")
        else:
            st.warning("Não foi possível extrair um valor numérico da resposta da IA. Tente reformular sua solicitação.")
    except Exception as e:
        st.error(f"Erro ao processar a solicitação: {str(e)}")
        
def set_api_key():
    # Update the API key in session state
    if 'api_key_input_top' in st.session_state and st.session_state.api_key_input_top:
        st.session_state.openai_api_key = st.session_state.api_key_input_top
        st.success("Chave de API OpenAI atualizada com sucesso!")


st.title('Simulador - Conversão de Vendas')

# Bloco de descrição do app acima da chave API
with st.expander('Descrição do App', expanded = True):
    st.write('O objetivo principal deste app é simular a propensão de clientes a comprar um produto com base em um modelo de machine learning, permitindo ajuste de threshold, análise de resultados e uso de IA generativa para customização do ponto de corte.')

# Bloco da chave API OpenAI
st.markdown('#### 🔑 Chave de API OpenAI')
st.text_input(
    "Chave de API OpenAI:",
    type="password",
    key="api_key_input_top",
    value=st.session_state.openai_api_key,
    on_change=set_api_key,
    help="Insira sua chave de API da OpenAI para usar a funcionalidade de IA Generativa."
)
st.markdown("Obtenha sua chave em [OpenAI Platform](https://platform.openai.com/api-keys)")

st.sidebar.info('💡 Dica: Ajuste o threshold para calibrar a sensibilidade do modelo. Use os atalhos rápidos ou IA generativa para facilitar a configuração ideal.')
with st.sidebar:
    c1, c2 = st.columns([0.6, 0.4])
    c1.image('./images/logo_fiap.png', width = 100)
    c1.markdown('<span style="font-weight:bold; font-size:18px;">Auto ML - Fiap [v1]</span>', unsafe_allow_html=True)
    c2.write('')

    # database = st.selectbox('Fonte dos dados de entrada (X):', ('CSV', 'Online'))
    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal = True)
    # st.toggle('Fonte dos dados de entrada (X)')
    # st.checkbox('Fonte dos dados de entrada (X):', value = True)
    # st.selectbox('Fonte dos dados de entrada (X):', ['CSV', 'Online'])
    # st.multiselect('Fonte dos dados de entrada (X):', ['CSV', 'Online'])


    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

#Tela principal
if database == 'CSV':

    if file:
        #carregamento do CSV
        Xtest = pd.read_csv(file)

        #carregamento / instanciamento do modelo pkl
        mdl_rf = load_model('./pickle/pickle_rf_pycaret')

        # --- Start of PyCaret Cache Fix ---
        # Define a local cache directory within your project for PyCaret/joblib
        local_pycaret_cache_dir = os.path.join(os.getcwd(), ".pycaret_cache_temp") # Using a "hidden" folder convention

        try:
            # Create the local cache directory if it doesn't exist
            if not os.path.exists(local_pycaret_cache_dir):
                os.makedirs(local_pycaret_cache_dir, exist_ok=True)
            
            # Check if the loaded model (pipeline) has a 'memory' attribute
            if hasattr(mdl_rf, 'memory'):
                current_memory_setting = mdl_rf.memory
                
                # If memory is enabled (i.e., not None), we need to redirect it.
                # This covers cases where memory is a joblib.Memory object, a path string, or True.
                if current_memory_setting is not None:
                    # Obter informações da configuração original (para logging interno, não exibido na UI)
                    if isinstance(current_memory_setting, Memory):
                        original_location = current_memory_setting.location
                    elif isinstance(current_memory_setting, str):
                        original_location = current_memory_setting
                    else:
                        original_location = "N/A or Default"
            
                    # Apenas redireciona o cache sem exibir mensagens ao usuário
                    mdl_rf.memory = Memory(location=local_pycaret_cache_dir, verbose=0)
        except Exception as e:
            st.error(f"Error occurred during PyCaret cache redirection: {e}")
        # --- End of PyCaret Cache Fix ---

        # Predict do modelo
        ypred = predict_model(mdl_rf, data = Xtest, raw_score = True)

        # Treshold controls section (slider and text input)
        # These controls will affect "Predições" and "Analytics Detalhado" tabs
        with st.expander("⚙️ Ajustar Treshold de Predição", expanded=True):
            col_slider, col_text_display = st.columns([3, 2])

            with col_slider:
                st.slider(
                    'Arraste para definir o Treshold:',
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,  # Finer step for treshold
                    key='slider_widget_for_treshold', # Widget's own key
                    value=st.session_state.treshold,    # Controlled component
                    on_change=sync_treshold_from_slider,
                    help="Este valor (entre 0.0 e 1.0) é o ponto de corte para classificar um cliente como 'True' (convertido) ou 'False' (não convertido) com base no score de predição."
                )

            with col_text_display:
                st.text_input(
                    "Ou digite o Treshold (e pressione Enter):",
                    key='text_widget_for_treshold', # Widget's own key
                    value=f"{st.session_state.treshold:.2f}", # Display master value, formatted
                    on_change=sync_treshold_from_text,
                    help="Digite um valor entre 0.0 e 1.0. Ex: 0.65"
                )
            
            st.markdown(f"##### Treshold atual configurado: **{st.session_state.treshold:.2f}**")
            
        # Add IA Generativa section for threshold adjustment in a separate section
        st.markdown("---")
        st.markdown("### 🤖 Ajuste de Threshold via IA Generativa")
        st.markdown("Use linguagem natural para ajustar o threshold do modelo.")
        
        # Funções para preencher o texto E ajustar o threshold diretamente
        def set_conservador():
            st.session_state.ai_prompt_input = "Quero um modelo conservador."
            st.session_state.ultimo_atalho = "conservador"
            st.session_state.treshold = 0.70

        def set_precisao():
            st.session_state.ai_prompt_input = "Quero priorizar precisão."
            st.session_state.ultimo_atalho = "precisao"
            st.session_state.treshold = 0.80

        def set_recall():
            st.session_state.ai_prompt_input = "Quero priorizar recall."
            st.session_state.ultimo_atalho = "recall"
            st.session_state.treshold = 0.30

        def set_f1_score():
            st.session_state.ai_prompt_input = "Quero maximizar o F1-score."
            st.session_state.ultimo_atalho = "f1"
            st.session_state.treshold = 0.50

        def set_roc():
            st.session_state.ai_prompt_input = "Quero o threshold ótimo pela curva ROC."
            st.session_state.ultimo_atalho = "roc"
            st.session_state.treshold = 0.50

        def set_especificidade():
            st.session_state.ai_prompt_input = "Quero priorizar especificidade."
            st.session_state.ultimo_atalho = "especificidade"
            st.session_state.treshold = 0.80

        def set_sensibilidade():
            st.session_state.ai_prompt_input = "Quero priorizar sensibilidade."
            st.session_state.ultimo_atalho = "sensibilidade"
            st.session_state.treshold = 0.20

        def set_lucro():
            st.session_state.ai_prompt_input = "Quero maximizar lucro."
            st.session_state.ultimo_atalho = "lucro"
            st.session_state.treshold = 0.55

        def set_custos():
            st.session_state.ai_prompt_input = "Quero minimizar custos."
            st.session_state.ultimo_atalho = "custos"
            st.session_state.treshold = 0.45

            
        # Atalhos para conceitos de threshold (4 colunas, mais compacto)
        st.markdown('#### Atalhos rápidos para ajuste de threshold:')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Adiciona um emoji de check (✓) quando o atalho for selecionado
            conservador_text = 'Modelo conservador ✅' if st.session_state.ultimo_atalho == 'conservador' else 'Modelo conservador'
            precisao_text = 'Priorizar precisão ✅' if st.session_state.ultimo_atalho == 'precisao' else 'Priorizar precisão'
            

            st.button(conservador_text, help='Threshold alto (0.70). Menos falsos positivos, mais precisão.', on_click=set_conservador)
            st.button(precisao_text, help='Threshold muito alto (0.80). Quando errar é muito caro.', on_click=set_precisao)
        with col2:
            recall_text = 'Priorizar recall ✅' if st.session_state.ultimo_atalho == 'recall' else 'Priorizar recall'
            f1_text = 'Maximizar F1-score ✅' if st.session_state.ultimo_atalho == 'f1' else 'Maximizar F1-score'
            
            st.button(recall_text, help='Threshold baixo (0.30). Para não perder nenhum cliente potencial.', on_click=set_recall)
            st.button(f1_text, help='Equilíbrio entre precisão e recall (0.50).', on_click=set_f1_score)
        with col3:
            roc_text = 'Threshold ótimo ROC ✅' if st.session_state.ultimo_atalho == 'roc' else 'Threshold ótimo ROC'
            especificidade_text = 'Priorizar especificidade ✅' if st.session_state.ultimo_atalho == 'especificidade' else 'Priorizar especificidade'
            
            st.button(roc_text, help='Threshold que maximiza sensibilidade e especificidade (0.50).', on_click=set_roc)
            st.button(especificidade_text, help='Threshold alto (0.80). Evita falsos positivos.', on_click=set_especificidade)
        with col4:
            sensibilidade_text = 'Priorizar sensibilidade ✅' if st.session_state.ultimo_atalho == 'sensibilidade' else 'Priorizar sensibilidade'
            lucro_text = 'Maximizar lucro ✅' if st.session_state.ultimo_atalho == 'lucro' else 'Maximizar lucro'
            custos_text = 'Minimizar custos ✅' if st.session_state.ultimo_atalho == 'custos' else 'Minimizar custos'
            
            st.button(sensibilidade_text, help='Threshold baixo (0.20). Para não perder positivos.', on_click=set_sensibilidade)
            st.button(lucro_text, help='Threshold que maximiza retorno financeiro (0.55).', on_click=set_lucro)
            st.button(custos_text, help='Threshold que minimiza custos de erro (0.45).', on_click=set_custos)
        
        # Natural language input for threshold - altura reduzida para uma linha
        st.text_input(
            "Descreva como você gostaria de ajustar o threshold:",
            placeholder="Ex: 'Quero um modelo mais conservador', 'Priorizar recall', 'Threshold ótimo pela curva ROC'",
            key="ai_prompt_input"
        )
        
        st.button(
            "Processar solicitação",
            on_click=process_natural_language_treshold,
            disabled=not st.session_state.openai_api_key,
            help="Clique para processar sua solicitação em linguagem natural e ajustar o threshold."
        )
        
        if not st.session_state.openai_api_key:
            st.info("⚠️ Configure sua chave de API OpenAI para usar esta funcionalidade.")

        current_treshold = st.session_state.treshold # Use this for all logic that depends on the treshold

        # Create tabs for different views
        tab_csv, tab_pred, tab_analytics = st.tabs(["📄 Visualizar CSV Carregado", "🎯 Visualizar Predições", "📊 Analytics Detalhado"])

        with tab_csv:
            st.subheader('Amostra do CSV Carregado')
            # c1_csv, _ = st.columns([2,4]) # Keep it simple for now
            qtd_linhas = st.slider('Visualizar quantas linhas do CSV:', 
                                    min_value = 5, 
                                    max_value = Xtest.shape[0], 
                                    step = 10,
                                    value = 5,
                                    key='csv_view_rows_slider')
            st.dataframe(Xtest.head(qtd_linhas))

        with tab_pred:
            st.subheader('Resultados da Predição')
            
            col_metric1, col_metric2 = st.columns(2)
            qtd_true = ypred.loc[ypred['prediction_score_1'] > current_treshold].shape[0]
            col_metric1.metric('Qtd clientes True (Conversão)', value = qtd_true)
            col_metric2.metric('Qtd clientes False (Não Conversão)', value = len(ypred) - qtd_true)
            
            def color_pred(val):
                color = 'lightgreen' if val > current_treshold else 'lightcoral' # Adjusted colors
                return f'background-color: {color}'

            tipo_view = st.radio('Selecione a visualização dos resultados:', 
                                 ('Completo (features + predições)', 'Apenas Scores de Predição'), 
                                 horizontal=True, 
                                 key='pred_view_type_radio')
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = ypred[['prediction_score_1']].copy()

            if 'prediction_score_1' in df_view.columns:
                st.dataframe(df_view.style.applymap(color_pred, subset=['prediction_score_1']))
            else:
                st.dataframe(df_view) # Fallback if column is missing

            csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label = 'Download CSV',
                            data = csv,
                            file_name = 'Predicoes.csv',
                            mime = 'text/csv',
                            key='download_pred_csv_button')

        with tab_analytics:
            st.subheader(f"Análise Comparativa por Feature (Treshold: {current_treshold:.2f})")
            st.markdown("Comparação das distribuições de features entre clientes preditos como '0' (Não Conversão) e '1' (Conversão).")

            ypred_analytics = ypred.copy()
            ypred_analytics['final_prediction'] = (ypred_analytics['prediction_score_1'] > current_treshold).astype(int)

            original_features = Xtest.columns.tolist()
            # Identify all available numeric features from the uploaded data
            available_numeric_features = [col for col in original_features if pd.api.types.is_numeric_dtype(Xtest[col]) and col in ypred_analytics.columns]

            if not available_numeric_features:
                st.info("Nenhuma feature numérica encontrada no CSV para realizar a análise gráfica.")
            else:
                # Define default features for selection
                default_features_for_selection = ["Income", "Recency"]
                # Filter default features to only include those actually present in the data
                actual_default_selection = [f for f in default_features_for_selection if f in available_numeric_features]

                # Add a multiselect widget for feature selection
                selected_features_to_plot = st.multiselect(
                    "Selecione as features para visualizar na análise:",
                    options=available_numeric_features,
                    default=actual_default_selection,
                    key='analytics_feature_selector'
                )

                for feature in selected_features_to_plot: # Iterate over selected features
                    st.markdown(f"---")
                    st.markdown(f"#### Análise da Feature: `{feature}`")

                    # Check for sufficient data variability for plotting
                    if ypred_analytics[feature].nunique() < 2:
                        st.caption(f"A feature '{feature}' possui menos de 2 valores únicos. Gráficos podem não ser informativos.")
                    
                    # Ensure there are data points for both prediction classes if possible
                    # This is more for robust error handling with plotting libraries
                    counts_by_prediction = ypred_analytics.groupby('final_prediction')[feature].count()
                    if not (0 in counts_by_prediction and 1 in counts_by_prediction and counts_by_prediction[0] > 0 and counts_by_prediction[1] > 0):
                         st.caption(f"Não há dados suficientes para ambas as classes de predição (0 e 1) para a feature '{feature}' com o treshold atual. Gráficos podem ser limitados a uma classe ou não gerados.")

                    try:
                        # Boxplot
                        fig_box = px.box(ypred_analytics,
                                         x='final_prediction',
                                         y=feature,
                                         color='final_prediction',
                                         labels={'final_prediction': f'Predição (0=Não Converte, 1=Converte | Treshold={current_treshold:.2f})', feature: feature},
                                         title=f'Boxplot de {feature} por Predição',
                                         color_discrete_map={0: 'orangered', 1: 'mediumseagreen'})
                        st.plotly_chart(fig_box, use_container_width=True)

                        # Histogram
                        fig_hist = px.histogram(ypred_analytics,
                                                x=feature,
                                                color='final_prediction',
                                                barmode='overlay',
                                                marginal='rug',
                                                labels={'final_prediction': f'Predição (0=Não Converte, 1=Converte | Treshold={current_treshold:.2f})', feature: feature},
                                                title=f'Histograma de {feature} por Predição',
                                                color_discrete_map={0: 'orangered', 1: 'mediumseagreen'})
                        fig_hist.update_layout(bargap=0.1)
                        fig_hist.update_traces(opacity=0.75) # Adjust opacity for overlay
                        st.plotly_chart(fig_hist, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Não foi possível gerar gráficos para a feature '{feature}': {e}")
        
    else:
        st.warning('Arquivo CSV não foi carregado')
        # st.info('Arquivo CSV não foi carregado')
        # st.error('Arquivo CSV não foi carregado')
        # st.success('Arquivo CSV não foi carregado')

else:
    # --- Opção ONLINE: Cadastro manual de cliente ---
    st.subheader('Cadastro Manual de Cliente')
    st.markdown('Preencha os campos abaixo para simular a predição de um novo cliente:')

    # Carregue o modelo treinado usando PyCaret
    try:
        model = load_model('pickle/pickle_rf_pycaret')

        # --- Start of PyCaret Cache Fix for ONLINE mode ---
        # Define a local cache directory within your project for PyCaret/joblib
        # This should be the same as used in the CSV section for consistency
        local_pycaret_cache_dir = os.path.join(os.getcwd(), ".pycaret_cache_temp") 

        try:
            # Create the local cache directory if it doesn't exist
            if not os.path.exists(local_pycaret_cache_dir):
                os.makedirs(local_pycaret_cache_dir, exist_ok=True)
            
            # Check if the loaded model (pipeline) has a 'memory' attribute
            if hasattr(model, 'memory'):
                current_memory_setting = model.memory
                
                # If memory is enabled (i.e., not None), we need to redirect it.
                if current_memory_setting is not None:
                    # Apenas redireciona o cache sem exibir mensagens ao usuário
                    model.memory = Memory(location=local_pycaret_cache_dir, verbose=0)
                    # st.sidebar.info(f"Cache do modelo (online) redirecionado para: {local_pycaret_cache_dir}") # Optional: for debugging
        except Exception as e:
            # Log or display this error if needed, but don't let it stop the app
            st.sidebar.warning(f"Erro ao redirecionar cache do modelo (online): {e}")
        # --- End of PyCaret Cache Fix for ONLINE mode ---

        st.success("Modelo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.info("Verifique se o arquivo do modelo existe e está no formato correto.")
        model = None

    # Defina as features usadas no modelo (ajuste conforme seu modelo)
    features_online = [
        'Age', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', # Changed Year_Birth to Age
        'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', # ...
        'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', # ...
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', # ...
        'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
        'Complain', 'Time_Customer'
    ]

    # Campos de entrada para cada feature
    input_data = {}
    
    # Crie colunas para melhor organização do formulário
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dados Demográficos")
        input_data['Age'] = st.number_input('Idade:', min_value=18, max_value=120, value=40, key='online_Age') # Changed from Year_Birth to Age
        input_data['Education'] = st.selectbox('Nível de Educação:', ['Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'], key='online_Education')
        input_data['Marital_Status'] = st.selectbox('Estado Civil:', ['Single', 'Together', 'Married', 'Divorced', 'Widow'], key='online_Marital_Status')
        input_data['Income'] = st.number_input('Renda Anual ($):', min_value=0, value=50000, key='online_Income')
        input_data['Kidhome'] = st.number_input('Número de Crianças em Casa:', min_value=0, max_value=5, value=0, key='online_Kidhome')
        input_data['Teenhome'] = st.number_input('Número de Adolescentes em Casa:', min_value=0, max_value=5, value=0, key='online_Teenhome')
        input_data['Recency'] = st.number_input('Dias desde última compra:', min_value=0, max_value=100, value=10, key='online_Recency')
        input_data['Time_Customer'] = st.number_input('Tempo como Cliente (dias):', min_value=0, value=500, key='online_Time_Customer')
        input_data['Complain'] = st.selectbox('Reclamou nos últimos 2 anos?', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='online_Complain')
    
    with col2:
        st.markdown("### Padrões de Compra")
        input_data['MntWines'] = st.number_input('Gastos com Vinhos ($):', min_value=0, value=300, key='online_MntWines')
        input_data['MntFruits'] = st.number_input('Gastos com Frutas ($):', min_value=0, value=50, key='online_MntFruits')
        input_data['MntMeatProducts'] = st.number_input('Gastos com Carnes ($):', min_value=0, value=250, key='online_MntMeatProducts')
        input_data['MntFishProducts'] = st.number_input('Gastos com Peixes ($):', min_value=0, value=50, key='online_MntFishProducts')
        input_data['MntSweetProducts'] = st.number_input('Gastos com Doces ($):', min_value=0, value=30, key='online_MntSweetProducts')
        input_data['MntGoldProds'] = st.number_input('Gastos com Gold Products ($):', min_value=0, value=50, key='online_MntGoldProds')
        input_data['NumDealsPurchases'] = st.number_input('Número de Compras com Desconto:', min_value=0, value=2, key='online_NumDealsPurchases')
        input_data['NumWebPurchases'] = st.number_input('Número de Compras Web:', min_value=0, value=4, key='online_NumWebPurchases')
        input_data['NumCatalogPurchases'] = st.number_input('Número de Compras por Catálogo:', min_value=0, value=2, key='online_NumCatalogPurchases')
        input_data['NumStorePurchases'] = st.number_input('Número de Compras em Loja:', min_value=0, value=5, key='online_NumStorePurchases')
        input_data['NumWebVisitsMonth'] = st.number_input('Visitas ao Site por Mês:', min_value=0, value=5, key='online_NumWebVisitsMonth')
    
    st.markdown("### Campanhas Aceitas")
    camp_col1, camp_col2, camp_col3, camp_col4, camp_col5 = st.columns(5)
    with camp_col1:
        input_data['AcceptedCmp1'] = st.selectbox('Campanha 1:', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='online_AcceptedCmp1')
    with camp_col2:
        input_data['AcceptedCmp2'] = st.selectbox('Campanha 2:', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='online_AcceptedCmp2')
    with camp_col3:
        input_data['AcceptedCmp3'] = st.selectbox('Campanha 3:', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='online_AcceptedCmp3')
    with camp_col4:
        input_data['AcceptedCmp4'] = st.selectbox('Campanha 4:', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='online_AcceptedCmp4')
    with camp_col5:
        input_data['AcceptedCmp5'] = st.selectbox('Campanha 5:', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='online_AcceptedCmp5')

    # Botão para simular
    if st.button('Simular Predição', key='online_predict_button'):
        # Ensure df_online is created with columns in the exact order of features_online
        # This helps if the pipeline is sensitive to column order or for consistency.
        df_online = pd.DataFrame([input_data], columns=features_online)

        # Ajuste de tipos se necessário
        # Added 'Income' to the list of numeric columns for explicit conversion.
        numeric_cols_to_convert = [
            'Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
            'Complain', 'Time_Customer'
        ]
        for col in numeric_cols_to_convert:
            if col in df_online.columns:
                df_online[col] = pd.to_numeric(df_online[col], errors='coerce')
                

        # Score de predição usando PyCaret
        if model is not None:
            # Fazer predição com PyCaret
            # Explicitly set raw_score=True to get probability scores,
            # as the subsequent code expects 'prediction_score_1'.
            # Pass the df_online that has enforced column order and types.
            predictions = predict_model(model, data=df_online, raw_score=True)

            # Extrair o score de probabilidade
            if 'prediction_score_1' in predictions.columns:
                score = predictions['prediction_score_1'].values[0]
                threshold = st.session_state.treshold
                pred = int(score > threshold)
                
                # Exibição destacada
                if pred == 1:
                    st.success(f'🎯 Cliente propenso a comprar! (Score: {score:.2f} | Threshold: {threshold:.2f})')
                else:
                    st.error(f'❌ Cliente NÃO propenso a comprar. (Score: {score:.2f} | Threshold: {threshold:.2f})')

                # --- Analytics Section for Online Prediction ---
                with st.expander("📊 Detalhes da Predição e Dados de Entrada", expanded=True):
                    st.markdown("#### Resumo da Predição:")
                    
                    # Display score, threshold, and decision in columns for better layout
                    col_score, col_thresh, col_decision = st.columns(3)
                    with col_score:
                        st.metric(label="Score de Propensão", value=f"{score:.4f}")
                    with col_thresh:
                        st.metric(label="Threshold Aplicado", value=f"{threshold:.2f}")
                    with col_decision:
                        st.metric(label="Decisão Final", value="Comprar" if pred == 1 else "Não Comprar")

                    # Gauge Chart for Propensity Score
                    st.markdown("---")
                    st.markdown("##### Visualização do Score de Propensão:")
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Propensão de Compra", 'font': {'size': 20}},
                        delta = {'reference': threshold, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue" if pred == 0 else "blue"}, # Color of the main bar
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, threshold], 'color': 'rgba(255, 0, 0, 0.3)'}, # Light red for below threshold
                                {'range': [threshold, 1], 'color': 'rgba(0, 255, 0, 0.3)'}   # Light green for above threshold
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("#### Dados de Entrada Fornecidos:")
                    
                    # Define feature groups based on your input form structure
                    demographic_features = ['Age', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'Time_Customer', 'Complain']
                    purchase_pattern_features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
                    campaign_features = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

                    def format_and_display_features(title, feature_list):
                        st.markdown(f"##### {title}")
                        data_to_display = {}
                        for feature_name in feature_list:
                            if feature_name in input_data:
                                value = input_data[feature_name]
                                if feature_name == 'Complain' or feature_name.startswith('AcceptedCmp'):
                                    data_to_display[feature_name] = 'Sim' if value == 1 else 'Não'
                                else:
                                    data_to_display[feature_name] = value
                        if data_to_display:
                            s_display = pd.Series(data_to_display, name='Valor Fornecido')
                            st.dataframe(s_display)
                        else:
                            st.caption("Nenhum dado para esta categoria.")

                    format_and_display_features("Dados Demográficos", demographic_features)
                    format_and_display_features("Padrões de Compra", purchase_pattern_features)
                    format_and_display_features("Campanhas Aceitas", campaign_features)
            else:
                st.error("Erro: O modelo não retornou scores de probabilidade.")
        else:
            st.error("Não foi possível fazer a predição porque o modelo não foi carregado corretamente.")
