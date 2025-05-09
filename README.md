    # Simulador de Conversão de Vendas - Case Ifood

    Este projeto é uma aplicação Streamlit desenvolvida para simular e prever a conversão de vendas de clientes, baseado em um modelo de Machine Learning. Ele permite que o usuário carregue dados de clientes (via CSV), ajuste um treshold de predição e visualize tanto as predições individuais quanto análises comparativas entre os grupos de clientes preditos como convertidos e não convertidos.

    ## Funcionalidades Principais

    *   **Carregamento de Dados**: Permite o upload de um arquivo CSV contendo os dados dos clientes para predição.
    *   **Predição de Conversão**: Utiliza um modelo de Machine Learning (Random Forest treinado com PyCaret) para gerar um score de propensão à conversão.
    *   **Ajuste de Treshold**:
        *   Um slider visual para definir o ponto de corte (treshold) entre 0.0 e 1.0.
        *   Um campo de texto para definir/alterar o valor do treshold.
    *   **Visualização de Predições**: Mostra os scores de predição e a classificação final (0 ou 1) para cada cliente, com a opção de baixar os resultados.
    *   **Analytics Detalhado**: Uma aba dedicada exibe gráficos comparativos (boxplots e histogramas) das features dos clientes, segmentados pela predição final (0 ou 1). Estes gráficos se ajustam dinamicamente conforme o treshold é alterado.

    ## Estrutura do Projeto

    *   `app_aula.py`: Arquivo principal da aplicação Streamlit.
    *   `requirements.txt`: Lista de dependências Python do projeto.
    *   `pickle/pickle_rf_pycaret`: Modelo de Machine Learning serializado.
    *   `images/`: Pasta para imagens (ex: logo).   
    *   (Outros arquivos de dados ou notebooks de desenvolvimento, se houver)

    ## Instruções para Execução

    Siga os passos abaixo para configurar e rodar o projeto localmente.

    ### Pré-requisitos

    *   Python 3.9 ou superior (recomendado 3.10 ou 3.11 para melhor compatibilidade com todas as bibliotecas de `requirements.txt`).
    *   `pip` (gerenciador de pacotes Python).

    ### 1. Clone o Repositório (se aplicável) ou Baixe os Arquivos

    Certifique-se de que todos os arquivos do projeto, incluindo `app_aula.py`, a pasta `pickle` com o modelo, e `requirements.txt`, estejam no mesmo diretório raiz do projeto (ex: `c:\Users\anton\Downloads\deploy\deploy\`).

    ### 2. Crie e Ative um Ambiente Virtual (Recomendado)

    É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

    ```bash
    python -m venv venv
    ```

    Para ativar o ambiente virtual:
    *   No Windows (Git Bash ou MINGW64):
        ```bash
        source venv/Scripts/activate
         ```
    *   No Windows (Command Prompt ou PowerShell):
        ```bash
        .\venv\Scripts\activate
        ```
    *   No macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

    ### 3. Instale as Dependências

    Navegue até o diretório raiz do projeto (`c:\Users\anton\Downloads\deploy\deploy\`) no seu terminal (com o ambiente virtual ativado) e execute:

    ```bash
    pip install -r requirements.txt
    ```

    ### 4. Execute a Aplicação Streamlit

    Ainda no diretório raiz do projeto, execute o seguinte comando no terminal:

    ```bash
    streamlit run app_aula.py
    ```

    A aplicação deverá abrir automaticamente no seu navegador web padrão. Caso contrário, o terminal exibirá os endereços (Local URL e Network URL) onde a aplicação está sendo servida.

    ## Utilização

    1.  Acesse a aplicação no seu navegador.
    2.  Na barra lateral, utilize a opção "Carregar CSV" para fazer o upload do arquivo com os dados dos clientes.
    3.  Ajuste o "Treshold de Predição" usando o slider ou o campo de texto.
    4.  Navegue pelas abas:
        *   **Visualizar CSV Carregado**: Para inspecionar os dados de entrada.
        *   **Visualizar Predições**: Para ver os scores e as predições finais, e baixar os resultados.
        *   **Analytics Detalhado**: Para explorar as diferenças nas features entre os clientes preditos como "0" e "1".

    ---
    *Desenvolvido como parte do case Ifood.*