#!/usr/bin/env node

import cliProgress from 'cli-progress';
import { program } from 'commander';
import dotenv from 'dotenv';
import fs from 'fs';
import inquirer from "inquirer";
import StreamZip from 'node-stream-zip';
import os from 'os';
import path from 'path';
import util from 'util';
import { prompt } from './prompts/prompt.js';

dotenv.config();

const MAX_RETRIES = parseInt(process.env.MAX_RETRIES) || 5;
const RETRY_DELAY = parseInt(process.env.RETRY_DELAY) || 3000;

const tmpDir = path.join(os.homedir(), 'tmp', '.eliza');
const envPath = path.join(tmpDir, '.env');

if (!fs.existsSync(tmpDir)) {
  fs.mkdirSync(tmpDir, { recursive: true });
}
if (!fs.existsSync(envPath)) {
  fs.writeFileSync(envPath, '');
}

let basicUserInfo = "";

const logError = (message, error) => {
  console.error(`[${new Date().toISOString()}] ERROR: ${message}`);
  if (error) {
    console.error(util.inspect(error, { depth: null, colors: true }));
  }
};

const parseJsonFromMarkdown = (text) => {
  const jsonMatch = text.match(/```json\n([\s\S]*?)\n```/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[1]);
    } catch (error) {
      logError('Error parsing JSON from markdown:', error);
    }
  }
  return null;
};

const promptUser = async (question, defaultValue = '') => {
  // Add a newline before the prompt
  console.log();

  const { answer } = await inquirer.prompt([
    {
      type: 'input',
      name: 'answer',
      message: question,
      default: defaultValue,
    },
  ]);
  return answer;
};

const runChatCompletion = async (messages, useGrammar = false, model) => {
  if (model === 'openai') {
    const modelName = 'gpt-4o';
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: modelName,
        messages: messages,
      }),
    });

    // check for 429
    if (response.status === 429) {
      await new Promise(resolve => setTimeout(resolve, 30000));
      return runChatCompletion(messages, useGrammar, model);
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices[0].message.content.trim();
    const parsed = parseJsonFromMarkdown(content) || JSON.parse(content);
    return parsed;
  }
  else if (model === 'claude') {
    const modelName = 'claude-3-5-sonnet-20240620';
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.CLAUDE_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: modelName,
        max_tokens: 8192,
        temperature: 0,
        messages: [
          {
            role: "user",
            content: messages[0].content
          }
        ],
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      logError(`HTTP error! status: ${response.status}`, errorData);
      throw new Error(`Anthropic API request failed with status: ${response.status}`);
    }

    const data = await response.json();
    const content = data.content[0].text;
    const parsed = parseJsonFromMarkdown(content) || JSON.parse(content);
    return parsed;
  }
  else if (model === 'openrouter') {
    const modelName = 'deepseek/deepseek-r1';
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({
        model: modelName,
        messages: messages,
      }),
    });

    // check for 429
    if (response.status === 429) {
      await new Promise(resolve => setTimeout(resolve, 30000));
      return runChatCompletion(messages, useGrammar, model);
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    const content = data.choices[0].message.content.trim();
    const parsed = parseJsonFromMarkdown(content) || JSON.parse(content);
    return parsed;
  }
};


const retryWithExponentialBackoff = async (func, retries = MAX_RETRIES) => {
  try {
    return await func();
  } catch (error) {
    if (retries > 0) {
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * (MAX_RETRIES - retries + 1)));
      return retryWithExponentialBackoff(func, retries - 1);
    }
    throw error;
  }
};

const validateJson = (json) => {
  const requiredKeys = ['bio', 'lore', 'adjectives', 'topics', 'style', 'messageExamples', 'postExamples'];
  const styleKeys = ['all', 'chat', 'post'];

  return requiredKeys.every(key => key in json) &&
    'style' in json &&
    styleKeys.every(key => key in json.style);
};

const ensureLogDirectory = () => {
  const logDir = path.join(tmpDir, 'logs');
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
};

const writeCacheFile = (cacheDir, fileName, content) => {
  const filePath = path.join(cacheDir, fileName);
  fs.writeFileSync(filePath, JSON.stringify(content, null, 2));
};

const readCacheFile = (cacheDir, fileName) => {
  const filePath = path.join(cacheDir, fileName);
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  }
  return null;
};

const saveProjectCache = (archivePath, cache) => {
  const cacheDir = path.join(tmpDir, 'cache', path.basename(archivePath, '.zip'));
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }
  writeCacheFile(cacheDir, 'project_cache.json', cache);

  // Save the model type to the project's .env file
  const envPath = path.join(cacheDir, '.env');
  const envConfig = {
    MODEL_TYPE: cache.model,
  };
  fs.writeFileSync(envPath, Object.entries(envConfig).map(([key, value]) => `${key}=${value}`).join('\n'));
};

const loadProjectCache = (archivePath) => {
  const cacheDir = path.join(tmpDir, 'cache', path.basename(archivePath, '.zip'));
  const cache = readCacheFile(cacheDir, 'project_cache.json');

  // Load the model type from the project's .env file
  const envPath = path.join(cacheDir, '.env');
  if (fs.existsSync(envPath)) {
    const envConfig = dotenv.parse(fs.readFileSync(envPath));
    if (envConfig.MODEL_TYPE) {
      cache.model = envConfig.MODEL_TYPE;
    }
  }

  return cache;
};

const clearGenerationCache = (archivePath) => {
  const cacheDir = path.join(tmpDir, 'cache', path.basename(archivePath, '.zip'));
  const files = fs.readdirSync(cacheDir);
  files.forEach((file) => {
    if (file.startsWith('prompt_response_')) {
      fs.unlinkSync(path.join(cacheDir, file));
    }
  });
};

const extractInfo = async (accountData, chunk, chunkIndex, archivePath, model) => {
  const cacheDir = path.join(tmpDir, 'cache', path.basename(archivePath, '.zip'));

  const promptFileName = `prompt_${chunkIndex}.json`;
  const promptResponseFileName = `prompt_response_${chunkIndex}_${model}.json`;

  const cachedPrompt = readCacheFile(cacheDir, promptFileName);
  const cachedPromptResponse = readCacheFile(cacheDir, promptResponseFileName);

  const name = accountData[0].account.accountDisplayName;
  const username = accountData[0].account.username;

  if (cachedPrompt && cachedPromptResponse) {
    return cachedPromptResponse;
  }

  writeCacheFile(cacheDir, promptFileName, { prompt: prompt(name, username, basicUserInfo, chunk) });
  let result;
  let attempts = 0;
  const maxAttempts = 3;
  do {
    attempts++;
    try {
      result = await retryWithExponentialBackoff(() => runChatCompletion([{ role: 'user', content: prompt(name, username, basicUserInfo, chunk) }], true, model));
      validateJson(result)
    } catch (error) {
      console.error(`Error processing chunk ${chunkIndex}, attempt ${attempts}:`, error);
      if (attempts >= maxAttempts) throw error;
    }
  } while (!validateJson(result) && attempts < maxAttempts)

  if (!validateJson(result)) {
    console.error(`Failed to get valid result for chunk ${chunkIndex} after ${maxAttempts} attempts`);
    return null;
  }

  writeCacheFile(cacheDir, promptResponseFileName, result);

  return result;
};

const buildConversationThread = async (tweet, tweets, accountData) => {
  let thread = [];
  const visited = new Set();

  async function processThread(currentTweet) {
    if (!currentTweet) {
      return;
    }
    if (visited.has(currentTweet.id_str)) {
      return;
    }
    visited.add(currentTweet.id_str);
    thread.unshift(currentTweet);
    if (currentTweet.in_reply_to_status_id_str) {
      const replyToTweet = tweets.find(
        (t) => t.id_str === currentTweet.in_reply_to_status_id_str
      );
      await processThread(replyToTweet);
    }
  }

  await processThread(tweet);
  thread = [...new Set(thread)];
  thread.sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );

  const conversationText = thread
    .map((t) => {
      const post = [];
      post.push(`From: ${accountData[0].account.accountDisplayName} (@${accountData[0].account.username})`);
      post.push(`Tweet ID: ${t.id_str}`);
      if (t.in_reply_to_status_id_str) {
        post.push(`In Reply To: ${t.in_reply_to_status_id_str}`);
      }
      post.push(`Timestamp: ${new Date(t.created_at).toLocaleString()}`);
      post.push(`Content:`);
      post.push(t.full_text);
      post.push("---");
      return post.join("\n");
    })
    .join("\n\n");

  return conversationText;
};

const chunkText = async (tweets, accountData, archivePath) => {
  const chunks = [];

  const CHUNK_SIZE = 60000; // 50k tokens approx

  const cacheDir = path.join(tmpDir, 'cache', path.basename(archivePath, '.zip'));

  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  if (Array.isArray(tweets)) {
    for (let i = 0; i < tweets.length; i += 1000) {
      const tweetChunk = tweets.slice(i, i + 1000);
      const conversationThreads = await Promise.all(
        tweetChunk.map((tweet) => buildConversationThread(tweet, tweets, accountData))
      );

      let currentChunk = "";

      for (const thread of conversationThreads) {
        if (thread.length > CHUNK_SIZE) {
          chunks.push(thread);
          continue;
        }
        // if length of current push is > threshold, push it and clear it
        if (currentChunk.length + thread.length > CHUNK_SIZE) {
          chunks.push(currentChunk);
          currentChunk = "";
        }
        currentChunk += thread;
      }
      // if current chunk is not empty, push it
      if (currentChunk.length > 0) {
        chunks.push(currentChunk);
      }
    }
  } else {
    console.error('Error: tweets is not an array');
  }

  // Save the unchunked data to cache
  fs.writeFileSync(path.join(cacheDir, 'unchunked_data.json'), JSON.stringify({ tweets, accountData }));

  // Save the chunks to cache
  chunks.forEach((chunk, index) => {
    const json = JSON.stringify(chunk);
    fs.writeFileSync(path.join(cacheDir, `chunk_${index}.json`), json);
  });

  return chunks;
};

const combineAndDeduplicate = (results) => {
  if (results.length === 0) {
    return {
      bio: '',
      lore: [],
      adjectives: [],
      topics: [],
      style: {
        all: [],
        chat: [],
        post: [],
      },
      messageExamples: [],
      postExamples: [],
    };
  }

  const combined = {
    bio: results.flatMap(result => result.bio),
    lore: [...new Set(results.flatMap((result) => result?.lore || []))],
    adjectives: [...new Set(results.flatMap((result) => result?.adjectives || []))],
    topics: [...new Set(results.flatMap((result) => result?.topics || []))],
    style: {
      all: [...new Set(results.flatMap((result) => result?.style?.all || []))],
      chat: [...new Set(results.flatMap((result) => result?.style?.chat || []))],
      post: [...new Set(results.flatMap((result) => result?.style?.post || []))],
    },
    messageExamples: [...new Set(results.flatMap((result) => result?.messageExamples || []))],
    postExamples: [...new Set(results.flatMap((result) => result?.postExamples || []))],
  };
  return combined;
};

const readFileFromZip = async (zip, fileName) => {
  try {
    const buffer = await zip.entryData(fileName);
    const content = buffer.toString('utf8');
    return content;
  } catch (error) {
    logError(`Error reading file ${fileName} from zip:`, error);
    throw error;
  }
};

process.on('uncaughtException', (error) => {
  logError('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logError('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

program
  .option('--openai <api_key>', 'OpenAI API key')
  .option('--claude <api_key>', 'Claude API key')
  .option('--openrouter <api_key>', 'OpenRouter API key')
  .parse(process.argv);

const limitConcurrency = async (tasks, concurrencyLimit) => {
  const results = [];
  const runningTasks = new Set();
  const queue = [...tasks];

  const runNext = async () => {
    if (queue.length === 0) return;
    const task = queue.shift();
    runningTasks.add(task);
    try {
      results.push(await task());
    } catch (error) {
      results.push(null);
      logError('Error in concurrent task:', error);
    } finally {
      runningTasks.delete(task);
      await runNext();
    }
  };

  const initialTasks = Array(Math.min(concurrencyLimit, tasks.length))
    .fill()
    .map(() => runNext());

  await Promise.all(initialTasks);
  await Promise.all(Array.from(runningTasks));

  return results;
};

const saveApiKey = (model, apiKey) => {
  const envConfig = dotenv.parse(fs.readFileSync(envPath));
  envConfig[`${model.toUpperCase()}_API_KEY`] = apiKey;
  fs.writeFileSync(envPath, Object.entries(envConfig).map(([key, value]) => `${key}=${value}`).join('\n'));
};

const loadApiKey = (model) => {
  const envConfig = dotenv.parse(fs.readFileSync(envPath));
  return envConfig[`${model.toUpperCase()}_API_KEY`];
};

const getApiKey = async (model) => {
  const envKey = process.env[`${model.toUpperCase()}_API_KEY`];
  if (validateApiKey(envKey, model)) return envKey;

  const cachedKey = loadApiKey(model);
  if (validateApiKey(cachedKey, model)) return cachedKey;

  let newKey = '';
  while (!validateApiKey(newKey, model)) {
    newKey = await promptForApiKey(model);
  }
  saveApiKey(model, newKey);
  return newKey;
};

const validateApiKey = (apiKey, model) => {
  if (!apiKey) return false;

  if (model === 'openai') {
    return apiKey.trim().startsWith('sk-');
  } else if (model === 'claude') {
    return apiKey.trim().length > 0;
  } else if (model === 'openrouter') {
    return apiKey.trim().startsWith('sk-');
  }
  return false;
};

const promptForApiKey = async (model) => {
  return await promptUser(`Enter ${model.toUpperCase()} API key: `);
};


const resumeOrStartNewSession = async (projectCache, archivePath) => {
  if (projectCache.unfinishedSession) {
    const choice = await promptUser(
      'An unfinished session was found. Continue? (Y/n): ',
      'Y'
    );
    if (choice.toLowerCase() !== 'y') {
      projectCache.unfinishedSession = null;
      clearGenerationCache(archivePath);
    }
  }

  if (!projectCache.unfinishedSession) {
    projectCache.model = await promptUser('Select model (openai/claude/openrouter): ');
    projectCache.basicUserInfo = await promptUser('Enter additional user info that might help the summarizer (real name, nicknames and handles, age, past employment vs current, etc): ');
    projectCache.unfinishedSession = {
      currentChunk: 0,
      totalChunks: 0,
      completed: false
    };
  }

  return projectCache;
};

const safeExecute = async (func, errorMessage) => {
  try {
    return await func();
  } catch (error) {
    logError(errorMessage, error);
    throw error;
  }
};

const saveCharacterData = (character) => {
  fs.writeFileSync('character.json', JSON.stringify(character, null, 2));
  console.log('Character data saved to character.json');
};

const main = async () => {
  try {
    let archivePath = program.args[0];

    if (!archivePath) {
      archivePath = await promptUser('Please provide the path to your Twitter archive zip file:');
    }

    let projectCache = loadProjectCache(archivePath) || {};

    projectCache = await resumeOrStartNewSession(projectCache, archivePath);

    const apiKey = await getApiKey(projectCache.model);
    if (!apiKey) {
      throw new Error(`Failed to get a valid API key for ${projectCache.model}`);
    }
    process.env[`${projectCache.model.toUpperCase()}_API_KEY`] = apiKey;

    saveProjectCache(archivePath, projectCache);

    const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
    progressBar.start(projectCache.unfinishedSession.totalChunks || 100, projectCache.unfinishedSession.currentChunk || 0);

    await safeExecute(async () => {
      const zip = new StreamZip.async({ file: archivePath });

      try {
        const accountData = JSON.parse((await readFileFromZip(zip, 'data/account.js')).replace('window.YTD.account.part0 = ', ''));

        const tweets = JSON.parse((await readFileFromZip(zip, 'data/tweets.js')).replace('window.YTD.tweets.part0 = ', ''))
          .map((item) => item.tweet)
          .filter((tweet) => !tweet.retweeted);

        const chunks = await chunkText(tweets, accountData, archivePath);

        projectCache.unfinishedSession.totalChunks = chunks.length;
        progressBar.setTotal(chunks.length);
        const tasks = chunks.map((chunk, index) => async () => {
          if (index < projectCache.unfinishedSession.currentChunk) {
            return null; // Skip already processed chunks
          }
          const result = await extractInfo(accountData, chunk, index, archivePath, projectCache.model);
          projectCache.unfinishedSession.currentChunk = index + 1;
          progressBar.update(projectCache.unfinishedSession.currentChunk);
          saveProjectCache(archivePath, projectCache);
          return result;
        });
        const results = await limitConcurrency(tasks, 3); // Process 3 chunks concurrently

        const validResults = results.filter(result => result !== null);
        const combined = combineAndDeduplicate(validResults);

        const character = {
          name: accountData[0].account.accountDisplayName,
          ...combined,
        };

        saveCharacterData(character);

        return character;
      } finally {
        await zip.close();
      }
    }, 'Error generating character JSON');

    progressBar.stop();

    projectCache.unfinishedSession.completed = true;
    saveProjectCache(archivePath, projectCache);
    clearGenerationCache(archivePath);
  } catch (error) {
    console.error('Error during script execution:', error);
    process.exit(1);
  }
};

main();