# Characterfile

The goal of this project is to create a simple, easy-to-use format for generating and transmitting character files. You can use these character files out of the box with [Eliza](https://github.com/elizaOS/eliza) or other LLM agents.

## Getting Started - Generate A Characterfile From Your Twitter

1. Open Terminal. On Mac, you can press Command + Spacebar and search for "Terminal". If you're using Windows, use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
2. Type `npx tweets2character` and run it. If you get an error about npx not existing, you'll need to install Node.js
3. If you need to install node, you can do that by pasting `curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash` into your terminal to install Node Version Manager (nvm)
4. Once that runs, make a new terminal window (the old one will not have the new software linked) and run `nvm install node` followed by `nvm use node`
5. Now copy and paste `npx tweets2character` into your terminal again.
6. NOTE: You will need to get a [Claude](https://console.anthropic.com/settings/keys) or [OpenAI](https://platform.openai.com/api-keys) API key. Paste that in when prompted
7. You will need to get the path of your Twitter archive. If it's in your Downloads folder on a Mac, that's ~/Downloads/<name of archive>.zip
8. If everything is correct, you'll see a loading bar as the script processes your tweets and generates a character file. This will be output at character.json in the directory where you run `npx tweets2character`. If you run the command `cd` in the terminal before or after generating the file, you should see where you are.

## Schema

The JSON schema for the character file is [here](schema/character.schema.json). This also matches the expected format for [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling).

Typescript types for the character file are [here](examples/types.d.ts).

## Examples

### Example Character file
Basic example of a character file, with values that are instructional
[examples/example.character.json](examples/example.character.json)

### Basic Python Example
Read the example character file and print the contents
[examples/example.py](examples/example.py)

### Python Validation Example
Read the example character file and validate it against the JSON schema
[examples/validate.py](examples/validate.py)

### Basic JavaScript Example
Read the example character file and print the contents
[examples/example.mjs](examples/example.mjs)

### JavaScript Validation Example
Read the example character file and validate it against the JSON schema
[examples/validate.mjs](examples/validate.mjs)

# Scripts

You can use the scripts the generate a character file from your tweets, convert a folder of documents into a knowledge file, and add knowledge to your character file.

Most of these scripts require an OpenAI or Anthropic API key.

## tweets2character

Convert your twitter archive into a .character.json

First, download your Twitter archive here: https://help.x.com/en/managing-your-account/how-to-download-your-x-archive

You can run tweets2character directly from your command line with no downloads:

```sh
npx tweets2character
```

Note: you will need node.js installed. The easiest way is with [nvm](https://github.com/nvm-sh/nvm).

Then clone this repo and run these commands:

```sh
npm install
node scripts/tweets2character.js twitter-2024-07-22-aed6e84e05e7976f87480bc36686bd0fdfb3c96818c2eff2cebc4820477f4da3.zip # path to your zip archive
```

Note that the arguments are optional and will be prompted for if not provided.

## folder2knowledge

Convert a folder of images and videos into a .knowledge file which you can use with [Eliza](https://github.com/lalalune/eliza). Will convert text, markdown and PDF into normalized text in JSON format.

You can run folder2knowledge directly from your command line with no downloads:

```sh
npx folder2knowledge <path/to/folder>
```

```sh
npm install
node scripts/folder2knowledge.js path/to/folder # path to your folder
```

Note that the arguments are optional and will be prompted for if not provided.

## knowledge2character

Add knowledge to your .character file from a generated knowledge.json file.

You can run knowledge2character directly from your command line with no downloads:

```sh
npx knowledge2character <path/to/character.character> <path/to/knowledge.knowledge>
```

```sh
npm install
node scripts/knowledge2character.js path/to/character.character path/to/knowledge.knowledge # path to your character file and knowledge file
```

Note that the arguments are optional and will be prompted for if not provided.

## Chat Export Processing

Process WhatsApp chat exports to create character profiles.

You can run chats2character directly from your command line with no downloads:

npx chats2character -f path/to/chat.txt -u "Username"
npx chats2character -d path/to/chats/dir -u "John Doe"

Or if you have cloned the repo:

npm install
node scripts/chats2character.js -f path/to/chat.txt -u "Username"
node scripts/chats2character.js -d path/to/chats/dir -u "John Doe"

Options:
-u, --user           Target username as it appears in chats (use quotes for names with spaces)
-f, --file           Path to single chat export file
-d, --dir            Path to directory containing chat files
-i, --info           Path to JSON file containing additional user information
-l, --list           List all users found in chats
--openai [api_key]   Use OpenAI model (optionally provide API key)
--claude [api_key]   Use Claude model (default, optionally provide API key)

Examples:
# Provide API key directly:
npx chats2character -d whatsapp/chats --openai sk-...
npx chats2character -d whatsapp/chats --claude sk-...

# Use stored/cached API key:
npx chats2character -d whatsapp/chats --openai
npx chats2character -d whatsapp/chats --claude

The script will look for API keys in the following order:
1. Command line argument if provided
2. Environment variables (OPENAI_API_KEY or CLAUDE_API_KEY)
3. Cached keys in ~/.eliza/.env
4. Prompt for key if none found

Example user info file (info.txt):
The user is a mother of two, currently living in Madrid. She works as a high school teacher
and has been teaching mathematics for over 15 years. She's very active in the school's
parent association and often organizes educational events. In her free time, she enjoys
gardening and cooking traditional Spanish recipes.

The file should be a plain text file with descriptive information about the user. This
information helps provide context to better understand and analyze the chat messages.

The script will:
1. Extract messages from the specified user
2. Process content in chunks
3. Generate a character profile
4. Save results to character.json

Note: WhatsApp chat exports should be in .txt format with standard WhatsApp export formatting:
[timestamp] Username: message

For usernames with spaces, make sure to use quotes:
[timestamp] John Doe: message

# License

The license is the MIT license, with slight modifications so that users are not required to include the full license in their own software. See [LICENSE](LICENSE) for more details.
