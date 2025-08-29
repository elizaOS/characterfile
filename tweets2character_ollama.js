import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import StreamZip from 'node-stream-zip';
import util from 'util';
import fetch from 'node-fetch';
import cliProgress from 'cli-progress';

dotenv.config();

const MAX_RETRIES = parseInt(process.env.MAX_RETRIES) || 5;
const RETRY_DELAY = parseInt(process.env.RETRY_DELAY) || 3000;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3';
const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('************* process.argv')
console.log(process.argv)
console.log('*************')

const log = (message, obj = null) => {
  console.log(`[${new Date().toISOString()}] ${message}`);
  if (obj) {
    console.log(util.inspect(obj, { depth: null, colors: true }));
  }
};

const logError = (message, error) => {
  console.error(`[${new Date().toISOString()}] ERROR: ${message}`);
  if (error) {
    console.error(util.inspect(error, { depth: null, colors: true }));
    if (error.stack) {
      console.error('Stack trace:');
      console.error(error.stack);
    }
  }
};

const parseJsonFromMarkdown = (text) => {
  const jsonMatch = text.match(/```json\n([\s\S]*?)\n```/);
  if (jsonMatch) {
    try {
      // Remove any comments or extra whitespace
      const jsonString = jsonMatch[1].replace(/\/\/.*$/gm, '').trim();
      return JSON.parse(jsonString);
    } catch (error) {
      logError('Error parsing JSON from markdown:', error);
      logError('Raw JSON string:', jsonMatch[1]);
    }
  }
  // If no JSON block found, try to parse the entire text as JSON
  try {
    return JSON.parse(text);
  } catch (error) {
    logError('Error parsing entire text as JSON:', error);
    logError('Raw text:', text);
  }
  return null;
};

// Update the generateCharacterJson function to use the new extractInfoFromChunks
const generateCharacterJson = async (archivePath) => {
  log(`Starting character generation from archive: ${archivePath}`);
  const zip = new StreamZip.async({ file: archivePath });

  try {
    const entries = await zip.entries();

    log('Reading account data...');
    const accountData = JSON.parse((await readFileFromZip(zip, 'data/account.js')).replace('window.YTD.account.part0 = ', ''));
    log('Account data:', accountData);

    log('Reading tweets...');
    const tweets = JSON.parse((await readFileFromZip(zip, 'data/tweets.js')).replace('window.YTD.tweets.part0 = ', ''))
      .map((item) => item.tweet)
      .filter((tweet) => !tweet.retweeted);
    log(`Parsed ${tweets.length} tweets`);

    const chunks = await chunkText(tweets, accountData, archivePath);

    progressBar.start(chunks.length, 0);

    const results = [];
    for (let i = 0; i < chunks.length; i++) {
      try {
        const result = await extractInfo(chunks[i], i, archivePath);
        results.push(result);
        progressBar.update(i + 1);

        // Save intermediate results after each successful chunk processing
        savePartialResults(results, archivePath);
      } catch (error) {
        logError(`Error processing chunk ${i}:`, error);
        // Continue with the next chunk even if this one failed
      }
    }
    progressBar.stop();

    if (results.length === 0) {
      throw new Error('No valid results were obtained from any chunks');
    }

    const combined = combineAndDeduplicate(results);

    log('Generating message examples...');
    const messageExamples = await generateMessageExamples(tweets);

    log('Generating post examples...');
    const postExamples = await generatePostExamples(tweets);

    const character = {
      name: accountData[0].account.accountDisplayName,
      ...combined,
      messageExamples,
      postExamples,
    };


    log('Writing full.character.json...');
    fs.writeFileSync('full.character.json', JSON.stringify(character, null, 2));
    log('full.character.json generated successfully');

    log('Consolidating character information...');
    const finalCharacter = await consolidateCharacter(character);
    log('Consolidated character information:', finalCharacter);

    log('Writing final character.json...');
    fs.writeFileSync('character.json', JSON.stringify(finalCharacter, null, 2));
    log('character.json generated successfully');

    return finalCharacter;
  } catch (error) {
    logError('Error generating character.json:', error);
    throw error;
  } finally {
    await zip.close();
  }
};

const retryWithExponentialBackoff = async (func, retries = MAX_RETRIES) => {
  try {
    return await func();
  } catch (error) {
    if (retries > 0) {
      log(`Retrying... (${MAX_RETRIES - retries + 1}/${MAX_RETRIES})`);
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * (MAX_RETRIES - retries + 1)));
      return retryWithExponentialBackoff(func, retries - 1);
    }
    throw error;
  }
};

const savePartialResults = (results, archivePath) => {
  const partialCharacter = combineAndDeduplicate(results);
  const outputPath = path.join(path.dirname(archivePath), 'partial_character.json');
  fs.writeFileSync(outputPath, JSON.stringify(partialCharacter, null, 2));
  log(`Saved partial results to ${outputPath}`);
};

// Modify the runOllamaCompletion function
const runOllamaCompletion = async (prompt) => {
  log('Running Ollama completion...');
  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      prompt: prompt,
      stream: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  const content = data.response.trim();
  log('Raw Ollama response:', content);
  
  // Try to parse as JSON first
  try {
    const parsed = JSON.parse(content);
    log('Parsed JSON response:', parsed);
    return parsed;
  } catch (jsonError) {
    log('Failed to parse response as JSON, attempting to extract JSON from text');
    const parsed = parseJsonFromMarkdown(content);
    if (parsed) {
      log('Extracted and parsed JSON from response:', parsed);
      return parsed;
    } else {
      log('Failed to extract JSON from response');
      throw new Error('Failed to parse JSON from Ollama response');
    }
  }
};


const validateJson = (json) => {
  if (!json || typeof json !== 'object') {
    log('Invalid JSON structure:', json);
    return false;
  }
  const requiredKeys = ['bio', 'lore', 'adjectives', 'topics', 'style'];
  const styleKeys = ['all', 'chat', 'post'];

  const isValid = requiredKeys.every(key => key in json) &&
    'style' in json &&
    styleKeys.every(key => key in json.style);

  if (!isValid) {
    log('JSON validation failed. Missing required keys.');
    log('JSON structure:', json);
  }

  return isValid;
};

const ensureLogDirectory = () => {
  const logDir = path.join(__dirname, 'logs');
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir);
  }
};

const logToFile = (fileName, content) => {
  ensureLogDirectory();
  const logPath = path.join(__dirname, 'logs', fileName);
  fs.writeFileSync(logPath, content);
  log(`Logged to file: ${logPath}`);
};

const writeCacheFile = (cacheDir, fileName, content) => {
  fs.writeFileSync(path.join(cacheDir, fileName), JSON.stringify(content, null, 2));
};

const readCacheFile = (cacheDir, fileName) => {
  const filePath = path.join(cacheDir, fileName);
  if (fs.existsSync(filePath)) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  }
  return null;
};

const extractInfo = async (chunk, chunkIndex, archivePath) => {
  log(`Extracting information from chunk ${chunkIndex}...`);
  const cacheDir = path.join('cache', path.basename(archivePath, '.zip'));

  const promptFileName = `prompt_${chunkIndex}.json`;
  const promptResponseFileName = `prompt_response_${chunkIndex}.json`;

  const cachedPrompt = readCacheFile(cacheDir, promptFileName);
  const cachedPromptResponse = readCacheFile(cacheDir, promptResponseFileName);

  if (cachedPrompt && cachedPromptResponse) {
    log(`Loading cached prompt and response for chunk ${chunkIndex}...`);
    return cachedPromptResponse;
  }

  const prompt = `The following are tweets from the user:

${chunk}

Given the following tweets, extract the following information:

1. A brief bio for the user (1-2 paragraphs)
2. 5-10 interesting facts about the user (lore)
3. 3-5 adjectives that describe the user's posts
4. 3-5 frequently discussed topics
5. 3-5 stylistic directions for how the user speaks which are very specific to this user's writing style
6. 3-5 stylistic directions for how the user writes posts (post), specific to how the user writes and formats posts and presents information

BIO
The bio should be very specific to this user. Who they are, what they like and dislike, where they live or are from, what they care about, what they do for a living, relationship status, everything. Be as detailed as possible in building a profile of them.

LORE
Lore should be true facts about the user. They should be things that the user has stated about themselves or revealed in a confident tone indicating their veracity. Be very specific, and especially emphasize weird, interesting, or unusual facts.

ADJECTIVES
Adjectives should be specific and unique to this user. They should be so unique that you could pick out this user among their friends by the adjectives. Be honest and real, not flowery, very specific.

TOPICS
Topics should be specific and unique to this user. Very niche topics are good. Broad topics are bad. These should be topics the user is unequivocally interested in, even if they are one of a few people in the world who cares.

STYLE DIRECTIONS
Your style directions should be extremely specific and detailed-- only applicable to the specific nuances of how the user writes, not general directions or advice.
Remember, only pick out the things that are unique about this user's way of writing/speaking. We are not interested in the content of the tweets, but the style.

Be concise and to the point. No flowery language and avoid assistant-like language. Be honest, raw, not mean and not nice.

Respond with a JSON object containing the extracted information. Wrap the JSON in a markdown code block. Here's an example of the expected output format:

\`\`\`json
{
"bio": "Brief user bio here...",
"lore": [
    "Interesting fact 1",
    "Interesting fact 2",
    "Interesting fact 3",
    ...
],
"adjectives": [
    "Adjective 1",
    "Adjective 2",
    "Adjective 3",
    ...
],
"topics": [
    "Topic 1",
    "Topic 2",
    "Topic 3",
    ...
],
"style": {
    "all": [
    "Style direction 1",
    "Style direction 2",
    "Style direction 3",
    ...
    ],
    "chat": [
    "Chat style 1",
    "Chat style 2",
    "Chat style 3",
    ...
    ],
    "post": [
    "Post style 1",
    "Post style 2",
    "Post style 3",
    ...
    ]
}
}
  \`\`\``;

  writeCacheFile(cacheDir, promptFileName, { prompt });

  let result;
  let attempts = 0;
  const maxAttempts = 3;
  do {
    attempts++;
    try {
      console.log(`Running Ollama completion (attempt ${attempts})...`);
      result = await retryWithExponentialBackoff(() => runOllamaCompletion(prompt));
      console.log('Ollama result:', result);
    } catch (error) {
      logError(`Error in Ollama completion (attempt ${attempts}):`, error);
      if (attempts >= maxAttempts) {
        throw error;
      }
    }
  } while (!result || !validateJson(result) && attempts < maxAttempts);

  if (!result || !validateJson(result)) {
    throw new Error(`Failed to get valid JSON after ${maxAttempts} attempts`);
  }

  writeCacheFile(cacheDir, promptResponseFileName, result);

  return result;
};

const extractInfoFromChunks = async (chunks, archivePath) => {
  log('Extracting information from chunks...');

  const cacheDir = path.join('cache', path.basename(archivePath, '.zip'));
  const cachedResults = [];
  const tasks = [];

  for (let i = 0; i < chunks.length; i++) {
    const promptResponseFileName = `prompt_response_${i}.json`;
    const cachedPromptResponse = readCacheFile(cacheDir, promptResponseFileName);

    if (cachedPromptResponse) {
      log(`Loading cached result for chunk ${i}...`);
      cachedResults.push(cachedPromptResponse);
    } else {
      tasks.push(async () => {
        const result = await extractInfo(chunks[i], i, archivePath);
        return result;
      });
    }
  }

  const concurrencyLimit = 5; // Adjust this value based on your needs and local system capabilities
  const results = await limitConcurrency(tasks, concurrencyLimit);

  return [...cachedResults, ...results.filter((result) => result !== null)];
};

const buildConversationThread = async (tweet, tweets, accountData) => {
  let thread = [];
  const visited = new Set();

  async function processThread(currentTweet) {
    if (!currentTweet) {
      log("No current tweet found, skipping");
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
  log(`Chunking text...`);
  const chunks = [];

  const cacheDir = path.join('cache', path.basename(archivePath, '.zip'));
  if (!fs.existsSync(cacheDir)) {
    fs.mkdirSync(cacheDir, { recursive: true });
  }

  if (Array.isArray(tweets)) {
    for (let i = 0; i < tweets.length; i += 250) {
      const tweetChunk = tweets.slice(i, i + 250);
      const conversationThreads = await Promise.all(
        tweetChunk.map((tweet) => buildConversationThread(tweet, tweets, accountData))
      );
      const chunkText = conversationThreads.join('\n\n');
      chunks.push(chunkText);
    }
  } else {
    log('Error: tweets is not an array');
  }

  log(`Created ${chunks.length} chunks.`);

  // Save the unchunked data to cache
  fs.writeFileSync(path.join(cacheDir, 'unchunked_data.json'), JSON.stringify({ tweets, accountData }));

  // Save the chunks to cache
  chunks.forEach((chunk, index) => {
    fs.writeFileSync(path.join(cacheDir, `chunk_${index}.json`), JSON.stringify(chunk));
  });

  return chunks;
};

const combineAndDeduplicate = (results) => {
  log('Combining and deduplicating results...');

  if (results.length === 0) {
    log('Error: No results to combine and deduplicate');
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
    };
  }

  const combined = {
    bio: results[0]?.bio || '',
    lore: [...new Set(results.flatMap((result) => result?.lore || []))],
    adjectives: [...new Set(results.flatMap((result) => result?.adjectives || []))],
    topics: [...new Set(results.flatMap((result) => result?.topics || []))],
    style: {
      all: [...new Set(results.flatMap((result) => result?.style?.all || []))],
      chat: [...new Set(results.flatMap((result) => result?.style?.chat || []))],
      post: [...new Set(results.flatMap((result) => result?.style?.post || []))],
    },
  };
  return combined;
};

const generateMessageExamples = async (tweets) => {
  log('Generating message examples...');
  const prompt = `Given the following tweets, generate 7 message examples that represent typical conversations for this user. Each example should have 2-4 messages.

Tweets: ${JSON.stringify(tweets.slice(0, 250))}

Respond with a JSON array of message examples. Each example should be an array of message objects with 'user' and 'content' properties. Use '{{user1}}' for the other participants. Wrap the JSON in a markdown code block.`;

  return await retryWithExponentialBackoff(() => runOllamaCompletion(prompt));
};

const generatePostExamples = async (tweets) => {
  log('Generating post examples...');
  const prompt = `Given the following tweets, select 20 tweets that best represent the user's typical posts and personality.

Tweets: ${JSON.stringify(tweets.slice(0, 200))}

Respond with a JSON array of selected tweets. Wrap the JSON in a markdown code block.`;

  return await retryWithExponentialBackoff(() => runOllamaCompletion(prompt));
};

const consolidateCharacter = async (character) => {
  log('Consolidating character information...');
  const exampleCharacter = fs.readFileSync('example.json', 'utf8');
  const prompt = `Given the following extracted information and the example character JSON, create a final consolidated character.json file. Ensure that the output follows the structure of the example character JSON.

Example Character JSON:
${exampleCharacter}

Extracted Information:
${JSON.stringify(character, null, 2)}

Respond with a JSON object containing the consolidated character information. Wrap the JSON in a markdown code block.`;

  let result;
  do {
    result = await retryWithExponentialBackoff(() => runOllamaCompletion(prompt));
  } while (!validateJson(result));

  // Log the result
  log('Consolidated character result:', result);

  // Save the result to a file
  const date = new Date().toISOString().replace(/:/g, '-');
  logToFile(`${date}_consolidated_character.json`, JSON.stringify(result, null, 2));

  return result;
};

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

const readFileFromZip = async (zip, fileName) => {
  log(`Reading file from zip: ${fileName}`);
  try {
    const buffer = await zip.entryData(fileName);
    const content = buffer.toString('utf8');
    log(`Successfully read ${fileName}`);
    return content;
  } catch (error) {
    logError(`Error reading file ${fileName} from zip:`, error);
    throw error;
  }
};

// Main execution
const archivePath = process.argv[2];
console.log("archivePath")
console.log(archivePath)
if (!archivePath) {
  logError('Error: Please provide the path to the Twitter archive zip file.');
  process.exit(1);
}

(async () => {
  try {
    console.log("Starting character generation. This may take a while...");
    console.log("Intermediate results will be saved as the process runs.");
    const generatedCharacter = await generateCharacterJson(archivePath);
    log('Script execution completed successfully.');
    console.log("Final output is in 'character.json'. You can also check 'full.character.json' for more detailed results.");
    console.log("If the process was interrupted, check for 'partial_character.json' in the same directory as your archive.");
  } catch (error) {
    logError('Error during script execution:', error);
    console.log("Check for 'partial_character.json' in the same directory as your archive for any partial results.");
  }
})();
