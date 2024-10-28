#!/usr/bin/env python3

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 

import os
import mimetypes
import re

from argparse import ArgumentParser
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core import vectorstores
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from sys import stderr
from termcolor import colored
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from urllib.parse import urlparse
from termcolor import colored

def main():
    #
    # Parse Arguments
    #
    parser = ArgumentParser()
    parser.add_argument("-v", help="increase output verbosity", action="store_true")
    parser.add_argument("-m", type=str, help="select OpenAI model to use", default="gpt-3.5-turbo")
    args, paths = parser.parse_known_args()

    #
    # load LLM
    #
    llm = ChatOpenAI(model=args.m)
    if args.v:
        print(">>> Loaded LLM: %s" % llm, file=stderr)

    #
    # load documents
    #
    
    loaders = {
        "text": lambda file: TextLoader(file).load(),
        "application/pdf": lambda file: PyPDFLoader(file).load(),
        "url": lambda file: WebBaseLoader(file).load(),
    }

#    docs = PyPDFLoader(paths[0]).load()
    docs = []

    for path in paths:
        # check if url:
        if urlparse(path).scheme in ("http", "https"):
            if args.v:
                print(">>> Loading %s as %s" % (path, "url"), file=stderr)
            docs.extend(loaders["url"](path))

        # check if file exists:
        elif not os.path.exists(path):
            raise FileNotFoundError("%s not found" %  path)
        
        # detect filetype
        else:
            mimetype, _ = mimetypes.guess_type(path)
            if mimetype.startswith("text/"):
                mimetype = "text"

            if mimetype not in loaders:
                raise ValueError("Unsupported file type: %s" % mimetype)
            else:
                if args.v:
                    print(">>> Loading %s as %s" % (path, mimetype), file=stderr)
                docs.extend(loaders[mimetype](path))

    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if args.v:
        print(">>> Split %d documents into %d chunks" % (len(docs), len(splits)), file=stderr)

    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=APIKeys.openai))

    vectorstore = InMemoryVectorStore(embedding=OpenAIEmbeddings())
    vectorstore.add_documents(splits)
    if args.v:
        print(">>> Vectorized %d chunks" % len(splits), file=stderr)
            
    simple_retriever = vectorstore.as_retriever()
    retriever = MultiQueryRetriever.from_llm(retriever=simple_retriever, llm=llm)
    if args.v:
        print(">>> Created retriever", file=stderr)

    #
    # History Prompt
    #
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    if args.v:
        print(">>> Created history-aware retriever", file=stderr)

    #
    # Prompt
    #
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Answer as detailed and easy to understand as possible."
        "\n\n"
        "{context}"
    )
        
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if args.v:
        print(">>> Created RAG chain", file=stderr)

    #
    # Memory
    #

    # We then define a simple node that runs the `rag_chain`.
    # The `return` values of the node update the graph state, so here we just
    # update the chat history with the input message and response.
    def call_model(state: State):
        response = rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    # Our graph consists only of one node:
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Finally, we compile the graph with a checkpointer object.
    # This persists the state, in this case in memory.
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    if args.v:
        print(">>> Created app memory\n", file=stderr)
    #
    # Chat
    #
    config = {"configurable": {"thread_id": "abc123"}}

    while True:
        try:
            question = input(colored("Q: ", "yellow", attrs=["reverse"]))
        except EOFError:
            print()
            break

        print(colored("A: ", "green", attrs=["reverse"]), parse_markdown(app.invoke({"input": question},
    config=config)["answer"]), end="\n\n")

# We define a dict representing the state of the application.
# This state has the same input and output keys as `rag_chain`.
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

def parse_markdown(text):
    lines = text.splitlines()
    formatted_text = ""
    in_code_block = False

    for line in lines:
        # Check for code blocks
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue  # Skip the line with ```
        elif in_code_block:
            formatted_text += colored(line + "\n", "green")
            continue

        # Check for headers
        if line.startswith("# "):
            level = len(line) - len(line.lstrip("#"))
            header_text = line.lstrip("#").strip()
            formatted_text += colored(header_text, "blue", attrs=["bold", "underline"]) + "\n"
            continue
        
        if line.startswith("## "):
            level = len(line) - len(line.lstrip("#"))
            header_text = line.lstrip("#").strip()
            formatted_text += colored(header_text, "blue", attrs=["bold"]) + "\n"
            continue
        
        if line.startswith("### "):
            level = len(line) - len(line.lstrip("#"))
            header_text = line.lstrip("#").strip()
            formatted_text += colored(header_text, "cyan", attrs=["bold"]) + "\n"
            continue

        # Check for blockquotes
        if line.startswith(">"):
            quote_text = line.lstrip(">").strip()
            formatted_text += colored(quote_text, "yellow") + "\n"
            continue

        # Check for tables (rows separated by "|")
        if "|" in line:
            table_row = "\t| ".join(line.split("|")).strip()
            formatted_text += table_row + "\n"
            continue

        # Inline formatting for bold, italic, and code (keeping the symbols)
        # Bold (**text** or __text__)
        line = re.sub(r"[^\*_](\*\*|__)(.+?)(\*\*|__)[^\*_]", lambda m: colored(m.group(), attrs=["bold"]), line)
        # Italic (*text* or _text_)
        line = re.sub(r"[^\*_](\*|_)([^\*_].+?[^\*_])(\*|_)[^\*_]", lambda m: colored(m.group(), attrs=["underline"]), line)
        # Inline code (`code`)
        line = re.sub(r"[^\*_](`)(.+?)`[^\*_]", lambda m: colored(m.group() + "`", "green"), line)

        # List items (bullets and numbers)
        # Bulleted list
        line = re.sub(r"^(\s*[-*])\s", lambda m: colored(m.group(1), "cyan") + " ", line)
        # Numbered list
        line = re.sub(r"^(\s*\d+\.)\s", lambda m: colored(m.group(1), "cyan") + " ", line)

        # Add processed line to formatted text
        formatted_text += line + "\n"

    return formatted_text

if __name__ == "__main__":
    main()