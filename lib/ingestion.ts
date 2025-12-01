import * as fs from 'fs/promises';
import * as path from 'path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenAI } from '@google/genai';
import { ChromaClient } from 'chromadb-client';
import pdf from 'pdf-parse';


const DOCUMENTS_DIR = path.join(process.cwd(), 'documents');
const COLLECTION_NAME = 'kummatty_policies';
const EMBEDDING_MODEL = 'models/embedding-001';
const CHROMA_HOST = process.env.CHROMA_HOST || 'http://localhost:8000';
const BATCH_SIZE = 100;

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  throw new Error("GEMINI_API_KEY environment variable not set.");
}
const ai = new GoogleGenAI({ apiKey });
const chromaClient = new ChromaClient({ path: CHROMA_HOST });

async function getOrCreateCollection() {
  console.log(`Attempting to connect to Chroma at: ${CHROMA_HOST}`);
  
  try {
    const collection = await chromaClient.getOrCreateCollection({ name: COLLECTION_NAME });
    console.log(`Successfully connected and retrieved/created collection: ${COLLECTION_NAME}`);
    return collection;
  } catch (error) {
    console.error("Error connecting to ChromaDB:", error);
    throw new Error("Failed to connect to ChromaDB. Ensure the service is running at " + CHROMA_HOST);
  }
}

/**
 * @param filePath The path to the PDF file.
 * @returns The extracted text.
 */
async function extractTextFromPdf(filePath: string): Promise<string> {
  const dataBuffer = await fs.readFile(filePath);

  if (typeof pdf !== 'function') {
      throw new Error("PDF parsing failed: The 'pdf-parse' module did not export a callable function correctly.");
  }

  const data = await pdf(dataBuffer);

  return data.text.trim();
}


/**
 * @param chunks An array of text chunks.
 * @returns An array of embedding vectors.
 */
async function generateEmbeddings(allChunks: string[]): Promise<number[][]> {
  console.log(`Generating embeddings for ${allChunks.length} chunks using ${EMBEDDING_MODEL}...`);
  
  const totalChunks = allChunks.length;
  let allEmbeddings: number[][] = [];
  
  for (let i = 0; i < totalChunks; i += BATCH_SIZE) {
    const batch = allChunks.slice(i, i + BATCH_SIZE);
    const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
    console.log(`Processing batch ${batchNumber}/${Math.ceil(totalChunks / BATCH_SIZE)} (Size: ${batch.length})...`);
    
    try {
      const response = await ai.models.embedContent({
        model: EMBEDDING_MODEL,
        contents: batch.map(text => ({ parts: [{ text }] })),
      });

      if (!response.embeddings || response.embeddings.length === 0) {
        throw new Error(`Batch ${batchNumber} failed to return embeddings.`);
      }

      const embeddings = response.embeddings.map(e => e.values as number[]);
      allEmbeddings = allEmbeddings.concat(embeddings);
      
    } catch (e: any) {
      console.error(`Embedding API Error on batch ${batchNumber}:`, e.message);
      throw new Error("Failed to call Gemini Embedding API. Check API Key and network connection.");
    }
  }

  console.log('All embeddings generated successfully.');
  return allEmbeddings;
}

export async function ingestDocuments(): Promise<{ success: boolean; count: number; error?: string }> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", " ", ""],
  });

  try {
    const collection = await getOrCreateCollection();
    
    const files = (await fs.readdir(DOCUMENTS_DIR)).filter(file => file.endsWith('.pdf'));
    if (files.length === 0) {
      return { success: false, count: 0, error: `No PDF files found in the directory: ${DOCUMENTS_DIR}` };
    }
    
    let chunkCount = 0;
    
    for (const fileName of files) {
      const filePath = path.join(DOCUMENTS_DIR, fileName);
      console.log(`\n--- Processing file: ${fileName} ---`);
      
      const fullText = await extractTextFromPdf(filePath);
      
      if (fullText.length === 0) {
        console.warn(`[SKIPPING] File ${fileName} could not be parsed or contains no text. Check if it's a scanned PDF.`);
        continue; 
      }
      
      const chunks = await splitter.splitText(fullText);

      if (chunks.length === 0) {
        console.warn(`[SKIPPING] File ${fileName} yielded text, but no chunks were created. Skipping ingestion.`);
        continue;
      }
      
      const embeddings = await generateEmbeddings(chunks);
      
      const ids: string[] = [];
      const metadatas: { source: string, chunk_index: number }[] = [];
      
      for (let i = 0; i < chunks.length; i++) {
        ids.push(`${fileName}-${i}`);
        metadatas.push({ 
            source: fileName, 
            chunk_index: i 
        });
      }
      
      await collection.upsert({
        ids: ids,
        embeddings: embeddings,
        documents: chunks,
        metadatas: metadatas
      });
      
      chunkCount += chunks.length;
      console.log(`File ${fileName} successfully ingested. Total chunks added/updated: ${chunkCount}`);
    }
    
    return { success: true, count: chunkCount };
    
  } catch (e: any) {
    console.error("INGESTION FAILED:", e.message);
    return { success: false, count: 0, error: e.message };
  }
}