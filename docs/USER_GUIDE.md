# DocuStore Local Assistant — User Guide

Welcome! This guide explains how to use **DocuStore Local Assistant**, the desktop app that answers questions using **your own documents** — entirely on your computer, with no internet connection required for your files.

---

## What is DocuStore Local Assistant?

DocuStore Local Assistant is an AI assistant you talk to like a chatbot. What makes it special:

- **It searches your documents.** Upload PDFs and text files, and the assistant finds answers inside them.
- **It is private.** Your documents are processed and stored on your own computer. Nothing is sent to the cloud.
- **It shows its sources.** Every answer lists the documents it used, so you can check where the information came from.
- **It can also search the web** when a question needs up-to-date information (like news or current events).

---

## 1. Starting the App

1. Double-click the **DocuStore Local Assistant** icon on your desktop (or open it from the Start menu).
2. A **loading screen** appears with a progress bar. Wait until it says **"Ready!"** — this can take a minute or two the first time, while the app loads its AI models.
3. The main chat window then opens automatically.

> The AI model must be downloaded before you can chat. See **Section 6 – Managing AI Models**. If the app was already set up for you, you can skip ahead.

---

## 2. The Main Screen

![Main chat screen](docs/screenshots/01_chat_main.png)

The app has two tabs at the top: **Chat** and **Settings**.

| Area                                | What it does                                                                                             |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Chat history (left sidebar)**     | Lists your previous conversations. Click one to reopen it.                                               |
| **New Chat**                        | Starts a fresh conversation.                                                                             |
| **Message area (centre)**           | Shows your questions and the assistant's answers.                                                        |
| **Message box (bottom)**            | Type your question here.                                                                                 |
| **Search domains (above messages)** | If you have organised your documents into groups (see Section 7), you can limit the search to one group. |

The small icons inside the message box switch the assistant's options on and off — see **Section 8 – Options** for details.

---

## 3. Asking a Question

1. Click in the **message box** at the bottom of the screen.
2. Type your question in plain English, for example:
   _"What are the holiday entitlement rules in the employee handbook?"_
3. Press **Enter** (or click the **send** arrow).

**While you wait**, the app tells you what it is doing, so you always know it's working:

![Assistant showing status while working](docs/screenshots/02_status_generating.png)

- **Thinking...** — preparing your question.
- **Searching your documents...** — looking for relevant information in your files.
- **Searching the web...** — looking online for current information (only when needed).
- **Generating response...** — writing the answer.

The answer then appears in the message area. You can keep chatting — the assistant remembers the conversation.

> **Tip:** Answers on a normal computer can take a little while (usually 20–60 seconds). That's normal — the AI is working locally, not in the cloud.

---

## 4. Reading the Answers

- **Thinking Process** — when the assistant "thinks through" a question, it shows a grey **Thinking Process** section above the answer. Click the heading to expand or hide it.
- **Citations (sources)** — underneath the answer you'll see blue buttons such as _Employee Handbook (p. 5)_. These are the documents the assistant used.
  - Click a button to **open the source** (a document page, or a website for web results).
  - Use the **arrow button** next to it to preview a short snippet of the source text.

If the assistant can't find an answer in your documents, it will say so — it won't make one up.

---

## 5. Managing Your Conversations (Sidebar)

- **New Chat** — clears the screen and starts a fresh conversation. Your previous chats stay saved.
- **Chat history** — click any saved conversation in the sidebar to reopen it.
- **Delete** — hover over a conversation and click the **×** button to remove it.
- **Collapse** — use the arrow at the bottom of the sidebar to hide it and give the chat more room.

---

## 6. Managing AI Models

![Model Management in Settings](docs/screenshots/03_settings_models.png)

The app uses an **AI model** to write its answers. Different models balance speed and quality.

1. Open the **Settings** tab and find **Model Management**.
2. Each available model is shown in its own card:
   - **Active** — the model currently in use.
   - **Downloaded** — installed on your computer. Click **Activate** to switch to it.
   - **Not downloaded** — click **Download** to install it. A progress bar shows the download.
3. Under the list you'll see **GPU Acceleration**, which tells you whether the app can use your graphics card to speed up answers (this is automatic).

> **Recommendation:** start with a smaller model (e.g. **Llama 3.2 1B**) for faster answers, and switch to a larger model (e.g. **Granite 4.1 3B**) when you want more detailed answers. Larger models are slower but smarter.

---

## 7. Managing Your Documents

![Document Management in Settings](docs/screenshots/04_settings_documents.png)

### Uploading documents

1. Open the **Settings** tab and find **Document Management**.
2. **Drag and drop** PDF or text (`.txt`) files into the box — or click the box to browse and select files. You can add several at once.
3. Your files appear in the list below.
4. Choose which **domain** to put them in (see _Domains_ below). If you're not using domains, keep **General**.
5. Click **Process & Ingest**. A progress bar shows the work being done. When it finishes, the assistant can search inside your new files.

**Clear Uploads** removes the staged files from the list without adding them to the assistant's knowledge.

### Organising documents into domains

![Domains in Settings](docs/screenshots/05_settings_domains.png)

Domains are folders for grouping related documents — for example "HR Policies", "Projects", or "Training".

- **View files in a domain** — pick a domain from the list to see which documents it contains.
- **Create a domain** — type a name in the box and click **Create**.
- **Delete a domain** — click the **×** on a domain card to delete it and all its documents. **General** cannot be deleted.

Once you have domains set up, you can use the **Search domains** filter on the Chat screen to limit the assistant to a specific group of documents.

---

## 8. Options (the icons in the message box)

| Icon            | Name                | What it does                                                                                                                                                     |
| --------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stack of layers | **Document search** | Searches your uploaded documents when answering. If you turn this off, the assistant answers from general knowledge only. _(On by default.)_                     |
| Lightbulb       | **Thinking**        | Lets the assistant reason through a question before answering (shows the Thinking Process section). _(On by default.)_                                           |
| Globe           | **Web search**      | Lets the assistant search the internet for up-to-date information. When on, it will only search the web when the question needs current data. _(On by default.)_ |

Click an icon to turn it on or off. When an option is **on**, the icon is highlighted in blue.

---

## 9. Dark Mode

Click the **sun/moon icon** at the top-right of the window to switch between light and dark appearance. The app remembers your choice next time you open it.

---

## 10. Troubleshooting

| Problem                                          | What to do                                                                                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| The app won't load (stuck on the loading screen) | Wait a few minutes on the first launch. If it still doesn't load, close and reopen the app.                                                         |
| "No chat model loaded"                           | Open **Settings → Model Management** and **Download** then **Activate** a model.                                                                    |
| Answers take a long time                         | A larger model is installed. Either switch to a smaller one (Section 6) or be patient — local answers are slower than online ones.                  |
| The assistant says it can't find the answer      | Make sure your files were uploaded **and** processed with **Process & Ingest**, and that the **Document search** option is on.                      |
| Web search doesn't seem to work                  | Check the **web search (globe)** icon is highlighted. The assistant only searches the web when a question is about current, up-to-date information. |
| An answer looks wrong                            | Check the citations under the answer to verify the source, then ask a more specific question.                                                       |

---

## 11. Privacy & Data

- Your documents, uploaded files, and conversations are stored **only on your computer**.
- The assistant does **not** send your questions or documents to any online service.
- The only time an internet connection is used is when **Web search** is turned on and the assistant looks up current information, or when you **download an AI model** in Settings.
