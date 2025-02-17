#!/usr/bin/env node

import fs from 'fs';
import inquirer from 'inquirer';

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

    const character = readJsonFile(characterFilePath);
    const knowledge = readJsonFile(knowledgeFilePath);

    if (!character || !knowledge) {
      console.error('Invalid input files. Please provide valid JSON files for character and knowledge.');
      return;
    }

    if (!outputFilePath) {
      const characterName = character.name.replace(/\s/g, '_');
      outputFilePath = `${characterName}.knowledge.character.json`;
    }

    character.knowledge = knowledge.chunks;

    writeJsonFile(outputFilePath, character);

    console.log('Script execution completed successfully.');
  } catch (error) {
    console.error('Error during script execution:', error);
  }
};

main();