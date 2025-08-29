#!/usr/bin/env node

import { program } from 'commander';
import dotenv from 'dotenv';
import fs from 'fs';
import inquirer from "inquirer";
import os from 'os';
import path from 'path';
import { prompt_whatsapp } from './prompts/prompt-whatsapp.js';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import cliProgress from 'cli-progress';
import chalk from 'chalk';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const MAX_RETRIES = 3;
const REQUEST_DELAY = 1000; // Delay between requests in milliseconds

const tmpDir = path.join(os.homedir(), 'tmp', '.eliza');
const envPath = path.join(tmpDir, '.env');

if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
}
if (!fs.existsSync(envPath)) {
    fs.writeFileSync(envPath, '');
}

// Console styling helpers
const Logger = {
    info: (msg) => console.log(chalk.cyan(`ℹ️  ${msg}`)),
    success: (msg) => console.log(chalk.green(`✅ ${msg}`)),
    warn: (msg) => console.log(chalk.yellow(`⚠️  ${msg}`)),
    error: (msg) => console.log(chalk.red(`❌ ${msg}`)),
    debug: (msg) => console.log(chalk.dim(`🔍 ${msg}`)),
    section: (msg) => {
        console.log('\n' + chalk.bold.cyan(msg));
        console.log(chalk.dim('═'.repeat(50)));
    },
    logAfterProgress: (msg) => {
        console.log(); // Print a new line before the message
        console.log(chalk.cyan(`ℹ️  ${msg}`)); // Log the message
    }
};

/*
 * Main execution function that processes whatsapp chat data to create character profiles:
 *
 * 1. Command Line Interface
 *    - Handles command line arguments for target user, input files/directories
 *    - Provides options for model selection (OpenAI vs Claude)
 *
 * 2. Input Processing
 *    - Validates and resolves input paths
 *    - Prompts user for missing required information
 *    - Handles user selection from available options
 *
 * 3. Message Processing
 *    - Chunks messages to stay within API limits
 *    - Tracks progress with progress bar
 *    - Processes chunks through selected AI model
 *
 * 4. Output Generation
 *    - Combines and deduplicates results
 *    - Validates generated character data
 *    - Saves results to output directory
 *    - Maintains processing cache for resumability
 */
const main = async () => {
    try {
        Logger.section('🤖 WhatsApp Chat Character Generator');

        /*
         * Command Line Interface Setup
         * Configures available CLI options and parses arguments
         */
        program
            .option('-u, --user <user>', 'Target user name')
            .option('-f, --file <file>', 'Single chat file to process')
            .option('-d, --dir <directory>', 'Directory containing chat files')
            .option('-i, --info <file>', 'Path to JSON file containing user info')
            .option('-l, --list', 'List all users found in chats')
            .option('--openai [api_key]', 'Use OpenAI model (optionally provide API key)')
            .option('--claude [api_key]', 'Use Claude model (optionally provide API key)')
            .option('--google [api_key]', 'Use Google model (optionally provide API key)')
            .parse(process.argv);

        const options = program.opts();

        /*
         * Input Path Resolution
         * Validates and gets input path from options or user prompt
         */
        let inputPath = options.file || options.dir || './chats';
        if (!fs.existsSync(inputPath)) {
            inputPath = await promptUser('Enter the path to chat file or directory:');
        }

        /*
         * Target User Selection
         * Gets target user from options or prompts user to select
         */
        let targetUser = options.user;
        if (!targetUser) {
            const users = findUsers(inputPath);
            if (users.length === 0) {
                throw new Error('No users found in chat file(s)');
            }

            const { selectedUser } = await inquirer.prompt([{
                type: 'list',
                name: 'selectedUser',
                message: fs.statSync(inputPath).isDirectory()
                    ? 'Select the user to analyze from all chats:'
                    : `Select the user to analyze from ${path.basename(inputPath)}:`,
                choices: users
            }]);

            targetUser = selectedUser;
        }

        /*
         * Model Selection and API Key Configuration
         * Determines AI model to use and sets up API keys
         */
        let model;
        if (options.openai || options.claude || options.google) {
            model = options.openai ? 'openai' : options.claude ? 'claude' : 'google';

            if (options.openai && options.openai !== true) {
                process.env.OPENAI_API_KEY = options.openai;
            } else if (options.claude && options.claude !== true) {
                process.env.CLAUDE_API_KEY = options.claude;
            } else if (options.google && options.google !== true) {
                process.env.GOOGLE_API_KEY = options.google;
            }
        } else {
            const { selectedModel } = await inquirer.prompt([{
                type: 'list',
                name: 'selectedModel',
                message: 'Select the model to use:',
                choices: ['openai', 'claude', 'google'],
                default: 'openai'
            }]);
            model = selectedModel;
        }
        console.log('Model:', model);

        Logger.info(`Processing input path: ${inputPath}`);

        /*
         * Message Processing Setup
         * Creates output directories and processes input messages
         */
        const dirs = createOutputDirs(targetUser, inputPath);

        // Read and parse messages (no caching for local processing)
        const messages = await (fs.statSync(inputPath).isDirectory()
            ? processDirectory(inputPath, targetUser, dirs)
            : extractMessagesFromFile(inputPath, targetUser, dirs));

        Logger.info(`Total messages processed: ${messages.length}`);

        /*
         * Message Chunking
         * Splits messages into manageable chunks for API processing
         */
        const chunks = await chunkMessages(messages, targetUser, model, dirs);

        /*
         * Progress Tracking
         * Sets up progress bar and processes chunks with status updates
         */
        const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
        progressBar.start(chunks.length, 0);

        Logger.info('');
        const results = [];

        /*
         * Check if cached API responses exist and prompt user
         */
        const cacheDir = path.join(tmpDir, 'cache', normalizeFileName(targetUser));
        const hasCachedResponses = fs.existsSync(cacheDir) && fs.readdirSync(cacheDir).some(file => file.startsWith('prompt_response_'));

        let useCache = true;
        if (hasCachedResponses) {
            const choice = await promptUser('Cached API responses found. Do you want to use them? (Y/n): ', 'Y');
            useCache = choice.toLowerCase() === 'y';

            if (!useCache) {
                Logger.info('Clearing cached API responses...');
                fs.rmSync(cacheDir, { recursive: true, force: true });
            }
        }

        /*
         * Message Processing Loop
         *
         * This section processes each message chunk sequentially:
         * 1. Extracts character info from each chunk using AI model
         * 2. Tracks progress and updates cache after each chunk
         * 3. Adds delay between API calls to avoid rate limits
         */
        for (let i = 0; i < chunks.length; i++) {
            const promptResponseFile = path.join(cacheDir, `prompt_response_${i}_${model}.json`);

            // Check for cached API response and use it if allowed
            if (useCache && fs.existsSync(promptResponseFile)) {
                Logger.info(`Using cached response for chunk ${i}`);
                const cachedResult = JSON.parse(fs.readFileSync(promptResponseFile, 'utf8'));
                results.push(cachedResult);
            } else {
                // Call the API if no cached response exists or cache is disabled
                const result = await extractInfo(
                    chunks[i].messages,
                    targetUser,
                    options.info || '',
                    i,
                    model
                );

                // Save the API response to cache
                fs.mkdirSync(cacheDir, { recursive: true });
                fs.writeFileSync(promptResponseFile, JSON.stringify(result, null, 2));
                results.push(result);
            }

            // Update progress bar
            progressBar.update(i + 1);
            await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY)); // Add delay between API calls
        }

        progressBar.stop();

        /*
         * Results Processing and Storage
         * Combines results, validates output, and saves to filesystem
         */
        const character = {
            name: targetUser,
            ...combineAndDeduplicate(results)
        };

        if (validateJson(character)) {
            saveCharacterData(character, dirs);
            Logger.success(`Character data saved to: ${dirs.character}`);
        } else {
            Logger.error('Character data validation failed.');
        }

        // Print summary
        printSummary(dirs);
        Logger.success('Script execution completed successfully.');
        process.exit(0);
    } catch (error) {
        Logger.error(`Error during script execution: ${error.message}`);
        process.exit(1);
    }
};

// Reuse these functions from tweets2character.js
const parseJsonFromMarkdown = (text) => {
    const jsonMatch = text.match(/```json\n([\s\S]*?)\n```/);
    if (jsonMatch) {
        try {
            return JSON.parse(jsonMatch[1]);
        } catch (error) {
            Logger.error('Error parsing JSON from markdown:', error);
        }
    }
    return null;
};

/**
 * Validates the structure of generated character JSON.
 * Checks for required fields and style structure.
 *
 * @param {Object} json - Character data object to validate
 * @returns {boolean} True if valid, false otherwise
 */
const validateJson = (json) => {
    const requiredKeys = ['bio', 'lore', 'adjectives', 'topics', 'style', 'messageExamples', 'postExamples'];
    const styleKeys = ['all', 'chat', 'post'];

    try {
        const hasRequiredKeys = requiredKeys.every(key => key in json);
        const hasValidStyle = 'style' in json && styleKeys.every(key => key in json.style);

        if (!hasRequiredKeys || !hasValidStyle) {
            Logger.error('Invalid JSON structure:');
            if (!hasRequiredKeys) {
                Logger.error('Missing required keys:', requiredKeys.filter(key => !(key in json)));
            }
            if (!hasValidStyle) {
                Logger.error('Invalid style structure');
            }
            return false;
        }

        return true;
    } catch (error) {
        Logger.error(`JSON validation error: ${error.message}`);
        return false;
    }
};

/**
 * Retries an operation with exponential backoff and validation.
 *
 * @param {Function} operation - Async function to execute
 * @param {Function} validator - Function to validate the result
 * @param {number} maxAttempts - Maximum number of retry attempts
 * @returns {Promise<any>} Validated result if successful
 * @throws {Error} If operation fails after all attempts or validation never passes
 */
const retryOperation = async (operation, validator, maxAttempts = MAX_RETRIES) => {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            Logger.info(`Calling API, attempt ${attempt}/${maxAttempts}`);
            const result = await operation();

            if (validator(result)) {
                return result;
            }

            Logger.warn(`Validation failed on attempt ${attempt}`);
        } catch (error) {
            const isRateLimit = error.status === 429;
            const shouldRetry = attempt < maxAttempts;

            if (isRateLimit && shouldRetry) {
                const delay = REQUEST_DELAY * attempt;
                Logger.warn(`Rate limit hit, waiting ${delay / 1000}s... (attempt ${attempt}/${maxAttempts})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                continue;
            }

            if (!shouldRetry) throw error;

            Logger.error(`Error on attempt ${attempt}: ${error.message}`);
        }

        if (attempt < maxAttempts) {
            const delay = REQUEST_DELAY * Math.pow(2, attempt - 1);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

    throw new Error(`Operation failed after ${maxAttempts} attempts`);
};

// Update extractInfo to use the new retry mechanism
const extractInfo = async (data, targetUser, basicUserInfo, chunk, model) => {
    const cacheDir = path.join(tmpDir, 'cache', normalizeFileName(targetUser));
    const promptFileName = `prompt_${chunk}.json`;
    const promptResponseFileName = `prompt_response_${chunk}_${model}.json`;

    // Check cache first
    const cachedPrompt = readCacheFile(cacheDir, promptFileName);
    const cachedPromptResponse = readCacheFile(cacheDir, promptResponseFileName);

    if (cachedPrompt && cachedPromptResponse) {
        return cachedPromptResponse;
    }

    Logger.info(`Processing chunk ${chunk} for ${targetUser}`);
    const promptContent = prompt_whatsapp(targetUser, targetUser, basicUserInfo, data);
    writeCacheFile(cacheDir, promptFileName, { prompt: promptContent });

    try {
        const result = await retryOperation(
            async () => runChatCompletion([{
                role: 'user',
                content: promptContent
            }], true, model),
            validateJson
        );

        writeCacheFile(cacheDir, promptResponseFileName, result);
        return result;
    } catch (error) {
        Logger.error(`Failed to process chunk ${chunk}: ${error.message}`);
        throw error;
    }
};
/**
 * Runs chat completion using either OpenAI or Claude API.
 *
 * Steps:
 * 1. Validates model type (openai or claude)
 * 2. Makes API request to selected model
 * 3. Processes response and returns parsed JSON
 *
 * OpenAI flow:
 * - Creates OpenAI client with API key
 * - Makes completion request with gpt-4
 * - Returns parsed JSON response
 *
 * Claude flow:
 * - Makes POST request to Anthropic API
 * - Uses claude-3-sonnet model
 * - Parses markdown or JSON response
 *
 * @param {Array} messages - Array of message objects to send to API
 * @param {boolean} useGrammar - Whether to use grammar checking (unused)
 * @param {string} model - Model type ('openai' or 'claude')
 * @returns {Object} Parsed JSON response from API
 * @throws {Error} If API request fails
 */
const runChatCompletion = async (messages, useGrammar = false, model) => {
    if (model === 'openai') {
        const openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });

        const response = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: messages,
            response_format: { type: 'json_object' }
        });

        return JSON.parse(response.choices[0].message.content);
    } else if (model === 'claude') {
        const modelName = 'claude-3-sonnet-20240229';
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
            throw new Error(`Anthropic API request failed with status: ${response.status}`);
        }

        const data = await response.json();
        const content = data.content[0].text;
        return parseJsonFromMarkdown(content) || JSON.parse(content);
    } else if (model === 'google') {
        const jsonMessages = JSON.stringify(messages);
        const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
        const response = await ai.models.generateContent(
            {
                model: "gemini-2.0-flash",
                contents: [{role: "user", parts: [{text: jsonMessages}]}],
                config: {
                    response_mime_type: "application/json"
                }
            }
        );
        console.log(response);
        const content = response.candidates[0].content.parts[0].text;
        return parseJsonFromMarkdown(content) || JSON.parse(content);
    }
};

/**
 * Normalizes a filename by converting to lowercase and removing special characters.
 *
 * @param {string} name - The filename to normalize.
 * @returns {string} The normalized filename with only lowercase letters, numbers, underscores and hyphens.
 */
const normalizeFileName = (name) => {
    return name.toLowerCase()
        .replace(/\s+/g, '_')  // Replace spaces with underscores.
        .replace(/[^a-z0-9_-]/g, ''); // Remove any other special characters.
};

/**
 * Creates output directory structure for storing processed chat data.
 *
 * Creates a directory structure under ./output/ with the following subdirectories:
 * - root: Base directory named after normalized username and source
 * - processed: For storing processed message data
 * - analytics: For storing analysis results
 *
 * @param {string} targetUser - Username to create directories for.
 * @param {string} inputPath - Path to input file/directory being processed.
 * @returns {Object} Object containing paths to created directories:
 *   - root: Base output directory
 *   - processed: Directory for processed files
 *   - analytics: Directory for analytics files
 */
const createOutputDirs = (targetUser, inputPath) => {
    // Normalize the target user name for file/directory naming.
    const normalizedUser = normalizeFileName(targetUser);

    // Get base name for single file or directory.
    const sourceName = fs.statSync(inputPath).isDirectory()
        ? 'directory'
        : path.basename(inputPath, path.extname(inputPath));

    // Create a base directory name combining normalized user and source.
    const baseDirName = `${normalizedUser}_${normalizeFileName(sourceName)}`;
    const baseDir = path.join(process.cwd(), 'output', baseDirName);

    const dirs = {
        root: baseDir,
        processed: path.join(baseDir, 'processed'),
        analytics: path.join(baseDir, 'analytics'),
        character: path.join(baseDir, 'character'),
        raw: path.join(baseDir, 'raw')
    };

    // Create all directories.
    Object.values(dirs).forEach(dir => {
        fs.mkdirSync(dir, { recursive: true });
    });

    return dirs;
};

/**
 * Extracts messages from a chat file for a specific user.
 *
 * This function processes a chat log file and extracts messages for a target user,
 * filtering out media messages, deleted messages and edits. If previously processed
 * messages exist for the user, prompts whether to reprocess.
 *
 * @param {string} filePath - Path to the chat log file to process.
 * @param {string} targetUser - Username to extract messages for.
 * @param {string} outputDir - Directory to save the extracted messages.
 * @returns {Promise<Array<Object>>} Array of message objects containing:
 *   - timestamp {string} Timestamp of the message
 *   - message {string} Content of the message
 * @throws {Error} If file cannot be read or processed
 *
 * TODO: since we are adding the timestamp to the output folder,
 * we cant check if the chats are already generated before,
 * we should name it as the username or
 * find a way to validate if the chats are already generated before
 */
const extractMessagesFromFile = async (filePath, targetUser, dirs, skipSave = false) => {
    // Only check for existing messages if we're not in directory processing mode
    if (!skipSave) {
        const normalizedUser = normalizeFileName(targetUser);
        const userOutputPath = path.join(dirs.processed, `${normalizedUser}_messages.json`);
        if (fs.existsSync(userOutputPath)) {
            const { reprocess } = await inquirer.prompt([{
                type: 'confirm',
                name: 'reprocess',
                message: `Found existing processed messages for ${targetUser}. Would you like to process again?`,
                default: false
            }]);

            if (!reprocess) {
                Logger.info('Using existing processed messages.');
                return JSON.parse(fs.readFileSync(userOutputPath, 'utf-8'));
            }
        }
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.split('\n');
    const messages = [];
    const messageRegex = /\[(.*?)\] (.*?):(.*)/;
    const foundUsers = new Set();

    lines.forEach(line => {
        const match = line.match(messageRegex);
        if (match) {
            const [, timestamp, user, message] = match;
            foundUsers.add(user.trim());

            // Skip media messages, deleted messages and message edits
            if (message.includes('image omitted') ||
                message.includes('sticker omitted') ||
                message.includes('audio omitted') ||
                message.includes('video omitted') ||
                message.includes('document omitted') ||
                message.includes('This message was deleted') ||
                message.includes('This message was edited')) {
                return;
            }

            // Case insensitive comparison
            if (user.trim().toLowerCase() === targetUser.trim().toLowerCase()) {
                messages.push({
                    timestamp,
                    message: message.trim(),
                    sourceFile: path.basename(filePath)
                });
            }
        }
    });

    Logger.success(`Found ${messages.length} messages for ${targetUser} in ${path.basename(filePath)}`);

    // Only save individual file results if not in directory processing mode
    if (!skipSave) {
        const normalizedUser = normalizeFileName(targetUser);
        const userOutputPath = path.join(dirs.processed, `${normalizedUser}_messages.json`);
        fs.writeFileSync(userOutputPath, JSON.stringify(messages, null, 2));
        Logger.success(`Messages saved to: ${userOutputPath}`);

        const usersListPath = path.join(dirs.analytics, 'found_users.json'); // TODO: do this for directories too
        fs.writeFileSync(usersListPath, JSON.stringify(Array.from(foundUsers), null, 2));
        Logger.success(`Users list saved to: ${usersListPath}`);
    }

    return messages;
};

/*
 * Processes an array of messages to identify and track unique messages and duplicates.
 * This function performs several key operations:
 *
 * 1. Message Deduplication:
 *    - Uses a Set to track unique message content
 *    - Filters out duplicate messages while preserving first occurrence
 *    - Maintains original timestamp and message content
 *
 * 2. Duplicate Tracking:
 *    - Records metadata for duplicate messages including:
 *      - Timestamp of duplicate occurrence
 *      - Source file where duplicate was found
 *    - Maps duplicate messages to array of occurrences
 *
 * 3. Output Generation:
 *    - Returns object containing:
 *      - Array of unique messages with timestamps
 *      - Map of duplicate messages with occurrence details.
 *
 * @param {Array} messages - Array of message objects containing timestamp, message content and sourceFile.
 * @returns {Object} Object containing uniqueMessages array and duplicateInfo map.
 * @property {Array} uniqueMessages - Array of deduplicated messages with timestamps.
 * @property {Object} duplicateInfo - Map of duplicate messages to their occurrence details.
 */
const getUniqueMessages = (messages) => {
    // Create a Set to track unique messages
    const uniqueSet = new Set();
    const uniqueMessages = [];
    const duplicateInfo = new Map();

    messages.forEach(msg => {
        if (!uniqueSet.has(msg.message)) {
            // Add to unique set and messages array
            uniqueSet.add(msg.message);
            uniqueMessages.push({
                timestamp: msg.message.timestamp,
                message: msg.message
            });
        } else {
            // Track duplicate information
            if (!duplicateInfo.has(msg.message)) {
                duplicateInfo.set(msg.message, []);
            }
            duplicateInfo.get(msg.message).push({
                timestamp: msg.timestamp,
                sourceFile: msg.sourceFile
            });
        }
    });

    return {
        uniqueMessages,
        duplicateInfo: Object.fromEntries(duplicateInfo)
    };
};

/*
 * Processes a directory of text files to extract and analyze messages for a target user.
 * This function performs several key operations:
 *
 * 1. Message Collection:
 *    - Scans the directory for .txt files
 *    - Extracts messages from each file for the target user
 *    - Tracks source file information for each message
 *
 * 2. Message Processing:
 *    - Combines messages from all files into a single collection
 *    - Identifies and handles duplicate messages across files
 *    - Generates unique message set removing duplicates
 *
 * 3. Output Generation:
 *    - Saves combined raw messages with source tracking
 *    - Creates filtered unique message set
 *    - Generates duplicate message analysis
 *    - Produces processing metadata and statistics
 *
 * @param {string} directory - Path to the directory containing text files to process.
 * @param {string} targetUser - Username to extract messages for.
 * @param {Object} dirs - Object containing output directory paths.
 * @param {string} dirs.processed - Path to save processed message files.
 * @param {string} dirs.analytics - Path to save analytics files.
 * @returns {Array} Array of unique messages extracted from all files, each containing timestamp and message content.
 * @throws {Error} If directory cannot be read or files cannot be processed.
 */
const processDirectory = async (directory, targetUser, dirs) => {
    const files = fs.readdirSync(directory);
    let allMessages = [];
    const processedFiles = new Set();

    // First pass: collect all messages from each file
    for (const file of files) {
        if (file.endsWith('.txt')) {
            const filePath = path.join(directory, file);
            Logger.info(`Processing file: ${filePath}`);

            const messages = await extractMessagesFromFile(filePath, targetUser, dirs, true);
            if (messages && messages.length > 0) {
                // Add source file tracking to each message
                allMessages = allMessages.concat(messages.map(msg => ({
                    ...msg,
                    sourceFile: path.basename(filePath)
                })));
                processedFiles.add(path.basename(filePath));
            }
        }
    }

    // Normalize user name for file names
    const normalizedUser = normalizeFileName(targetUser);

    // Save combined messages with source information for traceability
    const combinedOutputPath = path.join(dirs.processed, `${normalizedUser}_combined_messages.json`);
    fs.writeFileSync(combinedOutputPath, JSON.stringify(allMessages, null, 2));

    // Generate and save unique messages, removing duplicates while preserving metadata
    const { uniqueMessages, duplicateInfo } = getUniqueMessages(allMessages);

    // Save deduplicated message set
    const uniqueOutputPath = path.join(dirs.processed, `${normalizedUser}_unique_messages.json`);
    fs.writeFileSync(uniqueOutputPath, JSON.stringify(uniqueMessages, null, 2));

    // Save duplicate analysis for message tracking and verification
    const duplicatesPath = path.join(dirs.analytics, 'duplicate_messages_info.json');
    fs.writeFileSync(duplicatesPath, JSON.stringify(duplicateInfo, null, 2));

    // Generate and save comprehensive processing statistics
    const processingMeta = {
        totalFiles: processedFiles.size,
        processedFiles: Array.from(processedFiles),
        totalMessages: allMessages.length,
        uniqueMessages: uniqueMessages.length,
        duplicateMessages: allMessages.length - uniqueMessages.length,
        timestamp: new Date().toISOString()
    };

    fs.writeFileSync(
        path.join(dirs.analytics, 'processing_metadata.json'),
        JSON.stringify(processingMeta, null, 2)
    );

    // Return deduplicated message set for further analysis
    return uniqueMessages;
};

/*
 * Loads user information from a JSON file at the specified path.
 *
 * @param {string} infoPath - Path to the JSON file containing user information.
 * @returns {string|null} The contents of the file as a UTF-8 string if successful, null if error occurs.
 * @throws {Error} If file cannot be read or is invalid.
 */
const loadUserInfo = (infoPath) => {
    try {
        return fs.readFileSync(infoPath, 'utf-8');
    } catch (error) {
        Logger.error(`Error loading user info from ${infoPath}:`, error);
        return null;
    }
};

// Add API key handling functions
const saveApiKey = (model, apiKey) => {
    const envConfig = dotenv.parse(fs.readFileSync(envPath));
    envConfig[`${model.toUpperCase()}_API_KEY`] = apiKey;
    fs.writeFileSync(envPath, Object.entries(envConfig).map(([key, value]) => `${key}=${value}`).join('\n'));
};

const loadApiKey = (model) => {
    const envConfig = dotenv.parse(fs.readFileSync(envPath));
    return envConfig[`${model.toUpperCase()}_API_KEY`];
};

const validateApiKey = (apiKey, model) => {
    if (!apiKey) return false;

    if (model === 'openai') {
        return apiKey.trim().startsWith('sk-');
    } else if (model === 'claude') {
        return apiKey.trim().length > 0;
    }
    return false;
};

const promptForApiKey = async (model) => {
    return await promptUser(`Enter ${model.toUpperCase()} API key: `);
};

/**
 * Finds all unique users from chat message files.
 *
 * Process:
 * 1. Creates a Set to store unique usernames
 * 2. Defines regex pattern to match message format "[timestamp] username: message"
 * 3. For each file:
 *    - Reads file content
 *    - Splits into lines
 *    - Extracts usernames using regex
 *    - Adds to Set
 * 4. Handles both single files and directories of files
 * 5. Returns array of unique usernames
 *
 * @param {string} inputPath - Path to chat log file or directory
 * @returns {string[]} Array of unique usernames found
 */
const findUsers = (inputPath) => {
    const users = new Set();
    const messageRegex = /\[(.*?)\] (.*?):(.*)/;

    const processFile = (filePath) => {
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.split('\n');

        lines.forEach(line => {
            const match = line.match(messageRegex);
            if (match) {
                const [, , user] = match;
                users.add(user.trim());
            }
        });
    };

    if (fs.statSync(inputPath).isDirectory()) {
        const files = fs.readdirSync(inputPath);
        files.forEach(file => {
            if (file.endsWith('.txt')) {
                processFile(path.join(inputPath, file));
            }
        });
    } else {
        processFile(inputPath);
    }

    return Array.from(users);
};
/**
 * Combines and deduplicates character information from an array of results.
 *
 * Steps:
 * 1. Check if the results array is empty.
 *    - If it is, return a default object with empty fields for bio, lore, adjectives, topics, style, messageExamples, and postExamples.
 * 2. Create a combined object that aggregates data from all results:
 *    - bio: Concatenates all bio entries from each result.
 *    - lore: Collects unique lore entries using a Set to avoid duplicates.
 *    - adjectives: Collects unique adjectives using a Set.
 *    - topics: Collects unique topics using a Set.
 *    - style:
 *      - all: Collects unique styles from the 'all' category using a Set.
 *      - chat: Collects unique styles from the 'chat' category using a Set.
 *      - post: Collects unique styles from the 'post' category using a Set.
 *    - messageExamples: Collects unique message examples using a Set.
 *    - postExamples: Collects unique post examples using a Set.
 *
 * @param {Array} results - Array of character information objects to combine and deduplicate.
 * @returns {Object} Combined character information with deduplicated fields.
 *
 */
const combineAndDeduplicate = (results) => {
    if (results.length === 0) {
        return {
            // Provide a fallback single-item bio if no results
            bio: ["No data available."],
            lore: [],
            // The additional fields you requested
            plugins: [],
            clients: [],
            modelProvider: "",
            settings: {
                secrets: {},
                voice: {
                    model: ""
                }
            },
            system: "",
            adjectives: [],
            topics: [],
            style: {
                all: [],
                chat: [],
                post: []
            },
            messageExamples: [],
            postExamples: [],


        };
    }
    // ----- 1) Keep ONLY ONE bio -----
    // We retrieve the first non-empty bio array from 'results'
    // and take its first element as the single bio entry.
    const firstBio = results.find(r => Array.isArray(r.bio) && r.bio.length > 0)?.bio[0] ?? "";

    return {
        bio: [firstBio],   // Single-item array for the biography
        // ----- 2) Deduplicate the other fields as before -----
        lore: [...new Set(results.flatMap(r => r.lore || []))],
        adjectives: [...new Set(results.flatMap(r => r.adjectives || []))],
        topics: [...new Set(results.flatMap(r => r.topics || []))],
        style: {
            all: [...new Set(results.flatMap(r => (r.style?.all) || []))],
            chat: [...new Set(results.flatMap(r => (r.style?.chat) || []))],
            post: [...new Set(results.flatMap(r => (r.style?.post) || []))]
        },
        messageExamples: [...new Set(results.flatMap(r => r.messageExamples || []))],
        postExamples: [...new Set(results.flatMap(r => r.postExamples || []))],

        // ----- 3) Add your extra attributes here -----
        plugins: [],
        clients: [],
        modelProvider: "",
        settings: {
            secrets: {},
            voice: {
                model: ""
            }
        },
        system: ""
    };
};

/**
 * Saves processed character data to filesystem.
 *
 * @param {Object} characterData - Processed character information
 * @param {Object} dirs - Directory paths for saving data
 * @returns {Object} Directory paths used
 */
const saveCharacterData = (characterData, dirs) => {
    Logger.info('Saving character data...');

    // Normalize filename and create path
    const normalizedName = normalizeFileName(characterData.name);
    const filePath = path.join(dirs.character, `${normalizedName}.character.json`);

    // Write character data
    fs.writeFileSync(filePath, JSON.stringify(characterData, null, 2));
    Logger.success(`Character data saved to: ${filePath}`);
    return dirs;
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/************************************************************************************************
 * Cache & Session Management
 * This section handles caching functionality for the application, including session management,
 * reading/writing cache files, and clearing cached data.
 ************************************************************************************************/

/*
 * Handles resuming an unfinished session or starting a new one.
 * - Checks for existing unfinished session in cache
 * - Prompts user whether to continue unfinished session
 * - Clears cache if user chooses to start fresh
 *
 * @param {Object} projectCache - Existing cache object if any
 * @param {string} inputPath - Path to input chat files
 * @param {Object} options - CLI options object
 * @param {string} model - Selected AI model (openai/claude)
 * @returns {Object} Updated project cache with session info
 */
const resumeOrStartNewSession = async (projectCache, inputPath, options, model) => {
    // First check if there's a cached session and handle it
    if (projectCache?.unfinishedSession) {
        const cacheDir = path.join(tmpDir, 'cache', path.basename(inputPath));
        Logger.info(`\nFound cached session in: ${cacheDir}`);
        Logger.info('Session progress:', {
            currentChunk: projectCache.unfinishedSession.currentChunk,
            totalChunks: projectCache.unfinishedSession.totalChunks,
            completed: projectCache.unfinishedSession.completed
        });

        // Prompt specifically about continuing with cached session
        const choice = await promptUser('Want to continue with the cached session? (Y/n): ', 'Y');
        if (choice.toLowerCase() === 'y') {
            Logger.success('Continuing with cached session...');
            return projectCache; // Return existing cache without modification
        }

        // If not continuing with cached session, check for existing results
        Logger.warn('Starting fresh session...');
        projectCache.unfinishedSession = null;
        clearGenerationCache(inputPath);
        Logger.success('Cleared cached session.');
    }

    // Rest of the existing function for starting a new session
    let userInfo = '';
    if (options.info) {
        const loadedInfo = loadUserInfo(options.info);
        if (loadedInfo) {
            userInfo = typeof loadedInfo === 'string' ? loadedInfo : JSON.stringify(loadedInfo, null, 2);
        }
    }

    // Prompt for user info file if not provided
    if (!userInfo) {
        const infoPath = await promptUser('Enter path to user info file (optional, press enter to skip):');
        if (infoPath && fs.existsSync(infoPath)) {
            const loadedInfo = loadUserInfo(infoPath);
            if (loadedInfo) {
                userInfo = typeof loadedInfo === 'string' ? loadedInfo : JSON.stringify(loadedInfo, null, 2);
            }
        }
    }

    // Fall back to manual user info input
    if (!userInfo) {
        userInfo = await promptUser('Enter additional user info that might help the summarizer:');
    }

    // Initialize new project cache
    projectCache = {
        model: model,
        basicUserInfo: userInfo,
        unfinishedSession: {
            currentChunk: 0,
            totalChunks: 0,
            completed: false
        }
    };

    Logger.info('New session initialized:');

    // // Set up API key for chosen model
    // const apiKey = await getApiKey(projectCache.model);
    // if (!apiKey) {
    //     throw new Error(`Failed to get a valid API key for ${projectCache.model}`);
    // }
    // process.env[`${projectCache.model.toUpperCase()}_API_KEY`] = apiKey;

    return projectCache;
};

/*
 * Reads and parses a JSON cache file from the cache directory.
 * - Takes cache directory path and filename
 * - Returns parsed JSON content or null if file doesn't exist
 */
const readCacheFile = (cacheDir, fileName) => {
    const filePath = path.join(cacheDir, fileName);
    if (fs.existsSync(filePath)) {
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    }
    return null;
};

/*
 * Clears prompt response cache files for a given input path.
 * - Removes all files starting with 'prompt_response_' in cache directory
 * - Used when restarting an unfinished session
 */
const clearGenerationCache = (inputPath) => {
    const cacheDir = path.join(tmpDir, 'cache', path.basename(inputPath));
    if (fs.existsSync(cacheDir)) {
        const files = fs.readdirSync(cacheDir);
        files.forEach((file) => {
            if (file.startsWith('prompt_response_')) {
                fs.unlinkSync(path.join(cacheDir, file));
            }
        });
    }
};

/*
 * Writes content to a cache file in JSON format.
 * - Creates cache directory if it doesn't exist
 * - Writes formatted JSON content to specified file
 */
const writeCacheFile = (cacheDir, fileName, content) => {
    // Create cache directory if it doesn't exist
    if (!fs.existsSync(cacheDir)) {
        fs.mkdirSync(cacheDir, { recursive: true });
    }
    const filePath = path.join(cacheDir, fileName);
    fs.writeFileSync(filePath, JSON.stringify(content, null, 2));
};

/*
 * CLI Management and User Prompting
 *
 * Handles command-line interface interactions and user prompts
 */

/*
 * Prompts user for input with optional default value.
 * - Uses inquirer to show interactive prompt
 * - Returns user's answer as string
 */
const promptUser = async (question, defaultValue = '') => {
    console.log();
    const { answer } = await inquirer.prompt([{
        type: 'input',
        name: 'answer',
        message: question,
        default: defaultValue,
    }]);
    return answer;
};

/**
 * Chunks WhatsApp messages into smaller segments for AI processing.
 *
 * Process:
 * 1. Initialize empty arrays and counters for chunking messages
 * 2. Set max token limit (4000) for Claude-3 compatibility
 * 3. Create metadata object with user info and message stats
 * 4. For each message:
 *    - Estimate token count (4 chars ≈ 1 token)
 *    - If adding message exceeds token limit:
 *      - Save current chunk with metadata
 *      - Start new chunk with current message
 *    - Otherwise add message to current chunk
 * 5. Save final chunk if any messages remain
 * 6. Log chunking results and return chunks array
 *
 * @param {Array} messages - Array of WhatsApp message objects to chunk
 * @param {string} targetUser - Username/identifier of the chat participant
 * @param {string} model - Selected AI model ('openai' or 'claude')
 * @param {Object} dirs - Object containing output directory paths
 * @returns {Promise<Array>} Array of chunks, each containing:
 *   - metadata: Object with user info and message stats
 *   - messages: Array of message objects for this chunk
 *   - messageCount: Number of messages in this chunk
 *
 * Each chunk is sized to fit within model token limits:
 * - Claude: 4000 tokens
 * - OpenAI: 4096 tokens
 *
 * Token estimation uses rough 4 characters = 1 token approximation.
 * Metadata is added to each chunk to provide context for AI processing.
 */
const chunkMessages = async (messages, targetUser, model, dirs) => {
    Logger.info(`Starting message chunking for ${targetUser}...`);
    const chunks = [];
    let currentChunk = [];
    let currentTokens = 0;

    // Set token limit based on the model
    const MAX_TOKENS = model === 'claude' ? 4000 : 4096; // Adjust as needed for OpenAI models

    // Add metadata to help AI understand context
    const chunkMetadata = {
        user: targetUser,
        totalMessages: messages.length,
        messageType: 'whatsapp_chat'
    };

    for (const msg of messages) {
        // Estimate tokens (rough approximation: 4 chars = 1 token)
        const msgTokens = Math.ceil(msg.message.length / 4);

        if (currentTokens + msgTokens > MAX_TOKENS) {
            // Add metadata to chunk before pushing
            chunks.push({
                metadata: chunkMetadata,
                messages: currentChunk,
                messageCount: currentChunk.length
            });
            currentChunk = [msg.message];
            currentTokens = msgTokens;
        } else {
            currentChunk.push(msg.message);
            currentTokens += msgTokens;
        }
    }

    // Push final chunk if not empty
    if (currentChunk.length > 0) {
        chunks.push({
            metadata: chunkMetadata,
            messages: currentChunk,
            messageCount: currentChunk.length
        });
    }

    // Write chunks to debug file in the output directory
    const debugPath = path.join(dirs.root, 'chunks-debug.json');
    fs.writeFileSync(debugPath, JSON.stringify(chunks, null, 2));
    Logger.info(`Created ${chunks.length} chunks from ${messages.length} messages`);
    return chunks;
};

/**
 * Loads project cache from the filesystem.
 * Contains project configuration and basic info like model selection and API keys.
 *
 * @param {string} inputPath - Path to input file/directory being processed
 * @returns {Object|null} Project cache object or null if no cache exists
 */
const loadProjectCache = (inputPath) => {
    Logger.info(`Loading project cache for input path: ${inputPath}`);

    const cacheDir = path.join(tmpDir, 'cache', path.basename(inputPath));
    const cacheFile = path.join(cacheDir, 'project.json');

    try {
        if (fs.existsSync(cacheFile)) {
            const cache = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));

            // Load environment variables from .env if it exists
            const envPath = path.join(cacheDir, '.env');
            if (fs.existsSync(envPath)) {
                const envConfig = dotenv.parse(fs.readFileSync(envPath));
                if (envConfig.MODEL_TYPE) {
                    cache.model = envConfig.MODEL_TYPE;
                }
            }
            return cache;
        }
    } catch (error) {
        Logger.error(`Error loading project cache: ${error.message}`);
    }

    return null;
};

/**
 * Saves project cache to filesystem.
 * Stores project configuration, session state, and basic info.
 *
 * @param {string} inputPath - Path to input file/directory being processed
 * @param {Object} cache - Project cache object to save
 */
const saveProjectCache = (inputPath, cache) => {
    Logger.debug('Saving project cache...');

    const cacheDir = path.join(tmpDir, 'cache', path.basename(inputPath));
    const cacheFile = path.join(cacheDir, 'project.json');

    try {
        // Create cache directory if it doesn't exist
        if (!fs.existsSync(cacheDir)) {
            fs.mkdirSync(cacheDir, { recursive: true });
        }

        // Save cache object
        fs.writeFileSync(cacheFile, JSON.stringify(cache, null, 2));

        // Save environment variables if they exist
        if (cache.model) {
            const envPath = path.join(cacheDir, '.env');
            fs.writeFileSync(envPath, `MODEL_TYPE=${cache.model}\n`);
        }
    } catch (error) {
        Logger.error(`Error saving project cache: ${error.message}`);
    }
};

// Function to load cached results from the temp directory
const loadCachedResults = (cacheDir) => {
    const results = [];
    let i = 0;
    let filePath;

    while (true) {
        filePath = path.join(cacheDir, `prompt_response_${i}_openai.json`);
        if (fs.existsSync(filePath)) {
            const cachedResult = JSON.parse(fs.readFileSync(filePath, 'utf8'));
            results.push(cachedResult);
            i++;
        } else {
            break; // No more files to load
        }
    }
    return results;
};

const printSummary = (dirs) => {
    console.log('\n📂 Summary of Processed Files:');
    console.log(`├── ${dirs.root}`);
    console.log(`│   ├── character`);
    console.log(`│   │   └── ${path.basename(dirs.character)}.character.json`);
    console.log(`│   ├── raw`);
    console.log(`│   │   └── messages.json`);
    console.log(`│   ├── analytics`);
    console.log(`│   │   ├── duplicate_messages_info.json`);
    console.log(`│   │   └── processing_metadata.json`);
    console.log(`│   ├── processed`);
    console.log(`│   │   └── ${path.basename(dirs.processed)}_unique_messages.json`);
    console.log(`│   │   └── ${path.basename(dirs.processed)}_combined_messages.json`);
    console.log(`│   └── chunks-debug.json`);
    console.log('└── Process completed successfully.');
};

main();