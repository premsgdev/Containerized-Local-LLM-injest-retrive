import { GoogleGenAI } from '@google/genai';
import { ChromaClient } from 'chromadb-client'; 

const COLLECTION_NAME = 'kummatty_policies';
const EMBEDDING_MODEL = 'models/embedding-001';
const CHAT_MODEL = 'gemini-2.5-flash';
const CHROMA_HOST = process.env.CHROMA_HOST;
const TOP_K = 5;

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  throw new Error("GEMINI_API_KEY environment variable not set.");
}
const ai = new GoogleGenAI({ apiKey });
const chromaClient = new ChromaClient({ path: CHROMA_HOST });

interface ChromaQueryResult {
    ids: string[][];
    distances?: number[][];
    embeddings?: number[][];
    metadatas?: (object | null)[][];
    documents?: (string | null)[][];
}

/**
 * @param text The user's query string.
 * @returns The embedding vector.
 */
async function embedQuery(text: string): Promise<number[]> {
  const response = await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: [{ parts: [{ text }] }],
  });
  
  if (!response.embeddings || response.embeddings.length === 0) {
      throw new Error("Failed to generate embedding for the query.");
  }
  return response.embeddings[0].values as number[];
}

/**
 * @param queryEmbedding The vector representation of the user's query.
 * @returns The ChromaDB query result object.
 */
async function retrieveContext(queryEmbedding: number[]): Promise<ChromaQueryResult> {
    try {
        const collection = await chromaClient.getCollection({ name: COLLECTION_NAME });
        
        const results = await collection.query({
            queryEmbeddings: [queryEmbedding],
            nResults: TOP_K,
            include: ['documents', 'metadatas', 'distances'] as any
        });
        
        return results as unknown as ChromaQueryResult;
    } catch (error) {
        console.error("Error retrieving context from ChromaDB:", error);
        throw new Error("Failed to connect to ChromaDB or retrieve collection.");
    }
}

/**
 * @param queryResult The ChromaDB QueryResult containing documents and metadata.
 * @returns The structured context string for the LLM.
 */
function buildContextPrompt(queryResult: ChromaQueryResult): string {
    const documents = queryResult.documents?.[0] || [];
    const metadatas = queryResult.metadatas?.[0] || [];
    const distances = queryResult.distances?.[0] || [];
    
    let contextString = "You are a helpful and professional policy assistant. Your goal is to answer the user's query ONLY using the provided documents as context. If the answer is not in the documents, state clearly that you do not have enough information in the provided context.\n\n";
    contextString += "--- Retrieved Context Documents ---\n";

    documents.forEach((doc, index) => {
        const metadata = metadatas[index] as { source: string, chunk_index: number };
        const distance = distances[index];
        const documentText = doc || 'N/A';

        contextString += `\n[DOCUMENT ${index + 1}] Source: ${metadata.source}, Chunk Index: ${metadata.chunk_index}, Distance: ${distance.toFixed(4)}\n`;
        contextString += `${documentText}\n`;
    });

    contextString += "\n-----------------------------------\n";
    contextString += "Now, based on the context above, answer the user's question:\n";
    
    return contextString;
}

/**
 * @param userQuery The user's question.
 * @returns A streaming response from the Gemini model.
 */
export async function runRAGChat(userQuery: string): Promise<ReadableStream> {
    if (!userQuery) {
        throw new Error("User query cannot be empty.");
    }
    
    console.log(`Starting RAG process for query: "${userQuery.substring(0, 50)}..."`);

    const queryEmbedding = await embedQuery(userQuery);
    
    const queryResult = await retrieveContext(queryEmbedding);
    
    const totalRetrievedChunks = queryResult.documents?.[0]?.length || 0;
    if (totalRetrievedChunks === 0) {
        console.warn("No relevant chunks retrieved from ChromaDB.");
        const message = "I'm sorry, I couldn't find any relevant policy documents to answer your question in the database.";
        return new ReadableStream({
            start(controller) {
                controller.enqueue(new TextEncoder().encode(message));
                controller.close();
            }
        });
    }

    const contextPrompt = buildContextPrompt(queryResult);

    const responseStream = await ai.models.generateContentStream({
        model: CHAT_MODEL,
        contents: [
            {
                role: "user",
                parts: [{ text: contextPrompt + userQuery }]
            }
        ]
    });

    const encoder = new TextEncoder();
    return new ReadableStream({
        async start(controller) {
            for await (const chunk of responseStream) {
                const text = chunk.text;
                if (text) {
                    controller.enqueue(encoder.encode(text));
                }
            }
            controller.close();
        },
    });
}