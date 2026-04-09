import fs from 'fs';
import path from 'path';
import { NextResponse } from 'next/server';

export async function GET(request, { params }) {
  // Extract parameters in a way compatible with Next.js 14/15/16 App Router
  // Explicitly awaiting params is the safest pattern in newer Next.js versions
  const paramsData = await params;
  const name = paramsData.name;
  
  if (!name) return new NextResponse('Not found', { status: 404 });
  
  try {
    const filePath = path.join(process.cwd(), '../results', name);
    if (!fs.existsSync(filePath)) {
      return new NextResponse('Image not found', { status: 404 });
    }
    const buffer = fs.readFileSync(filePath);
    return new NextResponse(buffer, {
      headers: {
        'Content-Type': 'image/png',
        'Cache-Control': 'no-store, no-cache'
      }
    });
  } catch (error) {
    return new NextResponse('Internal server error', { status: 500 });
  }
}
