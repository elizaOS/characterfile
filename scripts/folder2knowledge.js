#!/usr/bin/env node

import pdf2md from '@opendocsg/pdf2md';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import readline from 'readline';

dotenv.config();

// The first argument from the command line is the starting path
const startingPath = process.argv[2];

const tmpDir = path.join(os.homedir(), 'tmp', '.eliza');
const envPath = path.join(tmpDir, '.env');

// Ensure the tmp directory and .env file exist
const ensureTmpDirAndEnv = async () => {
  await fs.mkdir(tmpDir, { recursive: true });
  if (!await fs.access(envPath).then(() => true).catch(() => false)) {
    await fs.writeFile(envPath, '');
  }
};

const saveApiKey = async (apiKey) => {
  const envConfig = dotenv.parse(await fs.readFile(envPath, 'utf-8'));
  envConfig.OPENAI_API_KEY = apiKey;
  await fs.writeFile(envPath, Object.entries(envConfig).map(([key, value]) => `${key}=${value}`).join('\n'));
};

const loadApiKey = async () => {
  const envConfig = dotenv.parse(await fs.readFile(envPath, 'utf-8'));
  return envConfig.OPENAI_API_KEY;
};

const validateApiKey = (apiKey) => {
  return apiKey && apiKey.trim().startsWith('sk-');
};

const promptForApiKey = () => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question('Enter your OpenAI API key: ', (answer) => {
      rl.close();
      resolve(answer);
    });
  });
};

const getApiKey = async () => {
  // Check process.env first
  if (validateApiKey(process.env.OPENAI_API_KEY)) {
    return process.env.OPENAI_API_KEY;
  }

  // Check cache in tmpdir
  const cachedKey = await loadApiKey();
  if (validateApiKey(cachedKey)) {
    return cachedKey;
  }

  // Prompt user if no valid key found
  const newKey = await promptForApiKey();
  if (validateApiKey(newKey)) {
    await saveApiKey(newKey);
    return newKey;
  } else {
    console.error('Invalid API key provided. Exiting.');
    process.exit(1);
  }
};

const processDocument = async (filePath) => {
  console.log(`Processing file: ${filePath}`);

  let content;
  const fileExtension = path.extname(filePath).toLowerCase();

  if (fileExtension === '.pdf') {
    const buffer = await fs.readFile(filePath);
    const uint8Array = new Uint8Array(buffer);
    content = await pdf2md(uint8Array);
  } else {
    content = await fs.readFile(filePath, 'utf8');
  }
  
  return content;
};

// Asynchronous function to recursively find files and process them
const findAndProcessFiles = async (dirPath) => {
  try {
    const filesAndDirectories = await fs.readdir(dirPath, {
      withFileTypes: true,
    });

    const contents = [];

    for (const dirent of filesAndDirectories) {
      const fullPath = path.join(dirPath, dirent.name);

      if (dirent.isDirectory()) {
        const _contents = await findAndProcessFiles(fullPath);
        contents.push(..._contents);
      } else if (dirent.isFile()) {
        const content = await processDocument(fullPath);
        contents.push(content);
      }
    }

    return contents;
  } catch (error) {
    console.error(`Error processing directory ${dirPath}: ${error}`);
    return [];
  }
};

const promptForPath = () => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question('Please enter a starting path: ', (answer) => {
      rl.close();
      resolve(answer);
    });
  });
};

// Main function to kick off the script
const main = async () => {
  try {
    await ensureTmpDirAndEnv();
    const apiKey = await getApiKey();
    process.env.OPENAI_API_KEY = apiKey;

    let path = startingPath;

    if (!path) {
      path = await promptForPath();
    }

    if (!path) {
      console.log('No starting path provided. Exiting.');
      return;
    }

    console.log(`Searching for files in: ${path}`);
    const contents = await findAndProcessFiles(path);

    // Save the output to knowledge.json
    await fs.writeFile('knowledge.json', JSON.stringify(contents, null, 2));

    console.log('Done processing files and saved memories to knowledge.json.');
  } catch (error) {
    console.error('Error during script execution:', error);
    process.exit(1);
  }
};

// Execute the main function
main();