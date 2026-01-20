import json
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

class RAGSystem:
    def __init__(self, corpus_path, model_name="llama3.1:8b"):
        print("=" * 60)
        print("INITIALIZING RAG SYSTEM")
        print("=" * 60)
        
        print(f"\n1. Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"  ✓ Loaded {len(corpus)} techniques")
        
        print("\n2. Converting to documents...")
        documents = []
        for item in corpus:
            content = f"ID: {item['id']}\nName: {item['name']}\nDesc: {item['description']}\nTactics: {', '.join(item['tactics'])}"
            doc = Document(page_content=content, metadata=item)
            documents.append(doc)
        
        print("\n3. Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.chunks = text_splitter.split_documents(documents)
        print(f"  ✓ Created {len(self.chunks)} chunks")
        
        print("\n4. Creating embeddings...")
        embeddings = OllamaEmbeddings(model=model_name)
        
        print("5. Building vector store...")
        self.vector_store = FAISS.from_documents(self.chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        print("\n6. Initializing LLM...")
        self.llm = OllamaLLM(model=model_name, temperature=0)
        
        template = """Analyze this network anomaly using MITRE ATT&CK.

MITRE Context:
{context}

Log:
{log_data}

Respond with JSON only:
{{"summary": "...", "mitre_id": "...", "mitre_name": "...", "severity": "..."}}"""
        
        self.prompt = PromptTemplate(template=template, input_variables=["context", "log_data"])
        self.rag_chain = (
            {"context": self.retriever, "log_data": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
        
        print("\n✓ RAG System ready!\n")
    
    def classify_threat(self, log_data):
        # Remove non-serializable fields
        clean_log = {}
        for key, value in log_data.items():
            if key == 'timestamp':
                continue  # Skip Firestore timestamp
            if isinstance(value, (str, int, float, bool, list, dict)):
                clean_log[key] = value
        
        log_str = json.dumps(clean_log, indent=2)
        
        try:
            response = self.rag_chain.invoke(log_str)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            return result
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "summary": "Potential network anomaly",
                "mitre_id": "T1071",
                "mitre_name": "Application Layer Protocol",
                "severity": "Medium"
            }
