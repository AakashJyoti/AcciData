import { createClient } from '@supabase/supabase-js';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

if (!SUPABASE_URL || !SUPABASE_KEY || !OPENAI_API_KEY) {
  throw new Error('Missing environment variables');
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

export class RAGService {
  private vectorStore: SupabaseVectorStore | null = null;
  private chain: ConversationalRetrievalQAChain | null = null;

  constructor() {
    this.initializeVectorStore();
  }

  private async initializeVectorStore() {
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: OPENAI_API_KEY,
    });

    this.vectorStore = await SupabaseVectorStore.fromExistingIndex(embeddings, {
      client: supabase,
      tableName: 'documents',
      queryName: 'match_documents',
    });

    const model = new ChatOpenAI({
      modelName: 'gpt-3.5-turbo',
      openAIApiKey: OPENAI_API_KEY,
      temperature: 0.7,
    });

    this.chain = ConversationalRetrievalQAChain.fromLLM(
      model,
      this.vectorStore.asRetriever(),
      {
        returnSourceDocuments: true,
        questionGeneratorTemplate: `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Chat History: {chat_history} Follow Up Input: {question}`,
        qaTemplate: `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}
Helpful Answer:`,
      }
    );
  }

  async addDocument(text: string, metadata: Record<string, any> = {}) {
    if (!this.vectorStore) {
      throw new Error('Vector store not initialized');
    }
    await this.vectorStore.addDocuments([{ pageContent: text, metadata }]);
  }

  async query(
    question: string,
    chatHistory: Array<[string, string]> = []
  ): Promise<{ text: string; sources: any[] }> {
    if (!this.chain) {
      throw new Error('Chain not initialized');
    }

    const response = await this.chain.call({
      question,
      chat_history: chatHistory,
    });

    return {
      text: response.text,
      sources: response.sourceDocuments || [],
    };
  }
}

export const ragService = new RAGService(); 