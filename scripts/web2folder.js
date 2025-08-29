#!/usr/bin/env node

import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import sanitizeFilename from 'sanitize-filename';
import os from 'os';
import readline from 'readline';

dotenv.config();

const FIRECRAWL_API_URL = 'https://api.firecrawl.dev/v1';

const tmpDir = path.join(os.homedir(), 'tmp', '.eliza');
const envPath = path.join(tmpDir, '.env');

const ensureTmpDirAndEnv = async () => {
  await fs.mkdir(tmpDir, { recursive: true });
  if (!await fs.access(envPath).then(() => true).catch(() => false)) {
    await fs.writeFile(envPath, '');
  }
};

const saveApiKey = async (apiKey) => {
  const envConfig = dotenv.parse(await fs.readFile(envPath, 'utf-8'));
  envConfig.FIRECRAWL_API_KEY = apiKey;
  await fs.writeFile(envPath, Object.entries(envConfig).map(([key, value]) => `${key}=${value}`).join('\n'));
};

const loadApiKey = async () => {
  const envConfig = dotenv.parse(await fs.readFile(envPath, 'utf-8'));
  return envConfig.FIRECRAWL_API_KEY;
};

const validateApiKey = (apiKey) => {
  return apiKey && apiKey.trim().startsWith('fc-');
};

const promptForApiKey = () => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question('Enter your Firecrawl API key: ', (answer) => {
      rl.close();
      resolve(answer);
    });
  });
};

const getApiKey = async () => {
  if (validateApiKey(process.env.FIRECRAWL_API_KEY)) {
    return process.env.FIRECRAWL_API_KEY;
  }

  const cachedKey = await loadApiKey();
  if (validateApiKey(cachedKey)) {
    return cachedKey;
  }

  const newKey = await promptForApiKey();
  if (validateApiKey(newKey)) {
    await saveApiKey(newKey);
    return newKey;
  } else {
    console.error('Invalid API key provided. Exiting.');
    process.exit(1);
  }
};

const promptForUrls = () => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question('Enter URLs (separated by spaces): ', (answer) => {
      rl.close();
      resolve(answer.split(' ').filter(url => url.trim()));
    });
  });
};

const scrapeUrl = async (url, apiKey) => {
  try {
    const response = await fetch(`${FIRECRAWL_API_URL}/scrape`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url })
    });

    const result = await response.json();
    
    if (!result.success) {
      throw new Error(`Failed to fetch ${url}: ${result.error}`);
    }

    const filename = sanitizeFilename(url.replace(/^https?:\/\//, '')) + '.md';
    
    return {
      url,
      content: result.data.markdown,
      filename
    };
  } catch (error) {
    console.error(`Error processing URL ${url}:`, error);
    return null;
  }
};

const saveToFolder = async (outputDir, webData) => {
  if (!webData) return;
  
  const outputPath = path.join(outputDir, webData.filename);
  await fs.writeFile(outputPath, webData.content, 'utf-8');
  console.log(`Saved ${webData.url} to ${outputPath}`);
};

const main = async () => {
  try {
    await ensureTmpDirAndEnv();
    const apiKey = await getApiKey();
    process.env.FIRECRAWL_API_KEY = apiKey;

    let urls = process.argv.slice(2);
    if (urls.length === 0) {
      urls = await promptForUrls();
      if (urls.length === 0) {
        console.error('No URLs provided. Exiting.');
        process.exit(1);
      }
    }

    const outputDir = path.join(process.cwd(), process.env.OUTPUT_DIR || 'web-content');
    await fs.mkdir(outputDir, { recursive: true });

    for (const url of urls) {
      const webData = await scrapeUrl(url, apiKey);
      if (webData) {
        await saveToFolder(outputDir, webData);
      }
    }

    console.log('Done processing web content.');
  } catch (error) {
    console.error('Error during script execution:', error);
    process.exit(1);
  }
};

main(); 