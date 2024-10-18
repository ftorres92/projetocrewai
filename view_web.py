from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
import openai
from crewai_tools import SerperDevTool
import os
import streamlit as st

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
serper_api_key = os.getenv('SERPER_API_KEY')

# Initialize tools and LLM
model_name="gpt-4o-mini"

# Tools
search_tool = SerperDevTool()

# Streamlit Interface
st.title("Agente Pesquisador e Criador de Conteúdo")
tema = st.text_input("Sobre qual tema gostaria de pesquisar?")

if st.button("Iniciar Pesquisa e Criar Postagem"):
    if tema:
        # 1. Agente Pesquisador de Notícias
        pesquisador = Agent(
            role=f"pesquisar as principais notícias sobre {tema}",
            goal=f"encontrar a mais recente notícia sobre o crescimento {tema}",
            backstory="""Você é um investigador incansável, 
            com um faro apurado para detectar as notícias mais relevantes e impactantes na vasta paisagem da internet. 
            Desde jovem, você sempre foi fascinado pelo fluxo contínuo de informações e como a narrativa das notícias pode moldar a opinião pública. 
            Agora, como Pesquisador de Notícias, você utiliza suas habilidades para vasculhar a web em busca das últimas manchetes,
            análises aprofundadas e dados confiáveis, garantindo que seus relatórios sejam uma referência de qualidade e precisão. 
            Sua dedicação à verdade e à clareza faz de você uma peça-chave na formação das estratégias de comunicação e na tomada de decisões informadas.""",
            memory=True,
            verbose=True,
            tools=[search_tool],
            llm=model_name
        )

        # 2. Agente Escritor de Roteiro
        escritor = Agent(
            role=f"Analisar o conteúdo encontrado e armazenado em notícias.md e criar uma postagem para o linkedin sobre {tema}",
            goal=f"Escrever uma postagem sobre o {tema} especificamente para ser utilizada em redes sociais",
            verbose=True,
            memory=True,
            backstory="""Você é um especialista em comunicação, com uma habilidade única para transformar informações complexas em conteúdos acessíveis e atraentes. 
            Com uma sólida experiência em marketing digital e redes sociais, você entende a importância de se conectar com a audiência certa através de mensagens claras e persuasivas. 
            Agora, como Criador de Conteúdo para Redes sociais, sua missão é analisar as últimas notícias, identificar os pontos mais relevantes e moldá-los em postagens que não apenas informem, mas também inspirem profissionais de diversas áreas. 
            Seu talento para criar narrativas que ressoam no ambiente corporativo faz de você o ponto central na comunicação de tendências e insights de mercado.""",
            llm=model_name
        )
      
         # 3. Agente Criador de conteúdo
        criador_conteudo = Agent(
            role='Creative Content Creator',
            goal='Transformar o texto fornecido pelo agente escritor em uma postagem envolvente para redes sociais, focada em captar o interesse e engajar o público.',
            backstory=(
                "Como Criador de Conteúdo em uma agência de marketing digital de ponta, "
                "você é especialista em criar narrativas que ressoam com o público nas redes sociais. "
                "Sua habilidade é transformar estratégias de marketing em histórias envolventes e visuais, "
                "capturando a atenção e inspirando ação."
            ),
            verbose=True
        )


        # Tarefas
        tarefa_pesquisa = Task(
            description=(
                f"Pesquisar informações detalhadas e relevantes sobre o {tema}, sempre fornecendo os respectivos links. "
                "Concentre-se em aspectos únicos e dados importantes que podem enriquecer a postagem, de modo que ela possa ser viral. "
                "Todo o texto deve estar em Português Brasil."
            ),
            expected_output=f'Um documento com as principais informações e dados sobre o {tema}, juntamento com os seus respectivos links.',
            tools=[search_tool],
            agent=pesquisador,
            output_file='noticias2.md'
        )

        tarefa_linkedin = Task(
            description=(
                "Analise as notícias mais recentes fornecidas pelo Pesquisador de Notícias e "
                "desenvolva uma postagem concisa e impactante para Redes sociais. "
                "A postagem deve destacar os pontos mais relevantes das notícias, "
                "ser informativa, envolvente e adequada ao público profissional da plataforma, sempre pensando em viralizar aos leitores."
            ),
            expected_output="Uma postagem finalizada para Redes sociais, pronta para publicação, "
                            "que resuma e destaque as notícias analisadas de forma clara e impactante, "
                            "com foco em gerar engajamento e discussão entre os profissionais da rede.",
            agent=escritor,
            output_file='postinicial.md'
        )
    
        criador_conteudo_task = Task(
            description=(
                "Transforme o texto a seguir em uma postagem envolvente para redes sociais. "
                "Foque em captar o interesse, humanizar a linguagem, criar uma conexão emocional e estimular o engajamento do público. "
                "Inclua metáforas, exemplos cotidianos, perguntas retóricas e emojis estratégicos. "
                "Conclua com uma 'call to action' que incentive interação."
            ),
            expected_output=(
                "Uma postagem completa e estruturada para redes sociais com linguagem humanizada, "
                "perguntas de engajamento, e uma chamada final para ação."
            ),
            agent=criador_conteudo,  
            output_file='postfinal.md'
        )

        # Criando a equipe (Crew) e executando
        crew = Crew(
            agents=[pesquisador, escritor, criador_conteudo],
            tasks=[tarefa_pesquisa, tarefa_linkedin, criador_conteudo_task],
            process=Process.sequential
        )

        result = crew.kickoff()
        st.success("Processo concluído!")
        with open('noticias2.md', 'r') as f:
            st.subheader("Notícias Encontradas:")
            st.markdown(f.read())
        with open('postinicial.md', 'r') as f:
            st.subheader("Postagem Inicial:")
            st.markdown(f.read())
        with open('postfinal.md', 'r') as f:
            st.subheader("Postagem Final")
            st.markdown(f.read())
    else:
        st.error("Por favor, insira um tema para iniciar a pesquisa.")