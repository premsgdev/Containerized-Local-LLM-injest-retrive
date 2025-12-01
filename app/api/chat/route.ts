import { runRAGChat } from '@/lib/retrieval';
import { NextRequest, NextResponse } from 'next/server';
export const runtime = 'nodejs';

export async function POST(req: NextRequest) {
  try {
    const { query } = await req.json();

    if (!query) {
      return NextResponse.json(
        { error: 'Missing query parameter in request body.' },
        { status: 400 }
      );
    }
    
    console.log(`[API] Received chat query: ${query}`);

    const stream = await runRAGChat(query);
    
    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked',
      },
      status: 200,
    });

  } catch (error: any) {
    console.error('Chat API Error:', error.message);
    
    return NextResponse.json(
      { error: 'An error occurred during RAG processing.', details: error.message },
      { status: 500 }
    );
  }
}