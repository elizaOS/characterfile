#!/usr/bin/env node

import fs from 'fs';
import inquirer from 'inquirer';

const createBasicCharacterFile = async (filePath) => {
  // See eliza types for reference: https://github.com/ai16z/eliza/blob/main/packages/core/src/core/types.ts
  const basicCharacter = {
    name: await promptUser('Enter character name:', 'New Character'),
    description: await promptUser('Enter character description:', 'A new character'),
    clients: [],
    modelProvider: 'llama_local',
    settings: {
      secrets: {},
      voice: {
        model: 'en_US-male-medium',
      },
    },
    traits: [],
    background: '',
    knowledge: {},
    messageExamples: [],
    postExamples: [],
    topics: [],
    style: {
      all: [],
      chat: [],
      post: [],
    },
    adjectives: [],
    nicknames: {},
    phrases: {},
  };

  writeJsonFile(filePath, basicCharacter);
  return basicCharacter;
};

const promptUser = async (question, defaultValue = '') => {
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

const readJsonFile = (filePath) => {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(fileContent);
  } catch (error) {
    console.error(`Error reading JSON file ${filePath}:`, error);
    return null;
  }
};

const writeJsonFile = (filePath, data) => {
  try {
    const jsonContent = JSON.stringify(data, null, 2);
    fs.writeFileSync(filePath, jsonContent, 'utf8');
    console.log(`Successfully wrote JSON file: ${filePath}`);
  } catch (error) {
    console.error(`Error writing JSON file ${filePath}:`, error);
  }
};

const main = async () => {
  try {
    let characterFilePath = process.argv[2];
    let knowledgeFilePath = process.argv[3];
    let outputFilePath = process.argv[4];

    if (!characterFilePath) {
      characterFilePath = await promptUser('Please provide the path to the character JSON file:', 'character.json');
    }

    if (!knowledgeFilePath) {
      knowledgeFilePath = await promptUser('Please provide the path to the knowledge JSON file:', 'knowledge.json');
    }

    let character;
    if (!fs.existsSync(characterFilePath)) {
      console.log(`Character file not found. Let's create one!`);
      character = await createBasicCharacterFile(characterFilePath);
    } else {
      character = readJsonFile(characterFilePath);
    }

    const knowledge = readJsonFile(knowledgeFilePath);

    if (!knowledge) {
      console.error('Invalid knowledge file. Please provide a valid JSON file for knowledge.');
      return;
    }

    if (!outputFilePath) {
      const characterName = character.name.replace(/\s/g, '_');
      outputFilePath = `${characterName}.knowledge.character.json`;
    }

    character.knowledge = knowledge;

    writeJsonFile(outputFilePath, character);

    console.log('Script execution completed successfully.');
  } catch (error) {
    console.error('Error during script execution:', error);
  }
};

main();