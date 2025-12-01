import { NextResponse } from 'next/server';
import { ingestDocuments } from '@/lib/ingestion';

export async function POST() {
  console.log("Ingestion request received...");
  
  try {
    const result = await ingestDocuments();
    
    if (result.success) {
      return NextResponse.json({ 
        message: `Ingestion successful. Total chunks added to ChromaDB: ${result.count}`,
        count: result.count
      }, { status: 200 });
    } else {
      return NextResponse.json({ 
        message: 'Ingestion failed.', 
        error: result.error 
      }, { status: 500 });
    }
    
  } catch (error: any) {
    console.error('API Ingestion Error:', error);
    return NextResponse.json({ 
      message: 'Internal server error during ingestion process.', 
      error: error.message 
    }, { status: 500 });
  }
}