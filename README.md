# 🤖 AI Customer Support Agent

An AI-powered customer support agent built using the **Customer Support on Twitter (TWCS)** dataset.

The system analyzes customer messages, identifies their intent, retrieves similar historical support conversations, generates a context-aware response, and decides whether the request can be handled automatically or should be escalated to a human agent.

The project focuses on **Comcast (`comcastcares`)**.

---

## 🚀 Key Features

- 💬 Customer conversation reconstruction
- 🧠 AI-based intent classification
- 🔍 Semantic search over historical conversations
- 📚 Retrieval-Augmented Generation (RAG)
- ✍️ Context-aware support response generation
- 🚨 Automatic escalation detection
- 📊 Intent classification evaluation
- 📈 Escalation evaluation
- 🧪 Golden evaluation dataset
- 🛡️ Privacy-aware response rules

---

## 🏗️ System Architecture

```text
                    Customer Message
                           │
                           ▼
                  ┌──────────────────┐
                  │  Intent Detection │
                  │   ML + Rules     │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Confidence Score │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Semantic Retrieval│
                  │      RAG         │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Historical Cases │
                  │     Comcast      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Response Generator│
                  │      Gemini      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Escalation Engine│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Final Decision  │
                  │ AUTO_HANDLE /    │
                  │ ESCALATE         │
                  └──────────────────┘