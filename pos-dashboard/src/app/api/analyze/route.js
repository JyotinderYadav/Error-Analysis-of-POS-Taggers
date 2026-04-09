import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
import util from 'util';

const execAsync = util.promisify(exec);

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file');
    const grokKey = formData.get('grokKey');

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    const backendDir = path.join(process.cwd(), '..'); 
    const uploadDir = path.join(backendDir, 'upload_data');
    
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    const filePath = path.join(uploadDir, 'custom.conllu');
    
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    fs.writeFileSync(filePath, buffer);

    console.log("Uploaded dataset saved to", filePath);

    const env = { ...process.env };
    // Force Node environment to see pip3 user packages
    env.PYTHONPATH = (env.PYTHONPATH ? env.PYTHONPATH + ':' : '') + '/Users/jyotinderyadav/Library/Python/3.9/lib/python/site-packages';
    
    if (grokKey && grokKey.trim() !== '') {
      env.GROK_API_KEY = grokKey.trim();
      console.log("Grok API key attached to environment.");
    }

    console.log("Executing backend pipeline...");
    
    // limit to 25 sentences so the xAI API and HuggingFace locally don't take ages
    const { stdout, stderr } = await execAsync(`/usr/bin/python3 main.py --test-file upload_data/custom.conllu --max-sentences 25`, {
      cwd: backendDir,
      env: env
    });

    console.log("Pipeline Finished.");

    return NextResponse.json({ success: true, message: 'Analysis returned successfully' });

  } catch (error) {
    console.error("Analyze Error:", error);
    return NextResponse.json({ error: error.message || 'Error occurred while processing dataset.' }, { status: 500 });
  }
}
