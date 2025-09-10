# 🤖 RAG Document Q&A Chatbot

A professional RAG (Retrieval-Augmented Generation) chatbot built with Streamlit, Google Gemini AI, and SERPAPI for document-based question answering with web search capabilities.

## ✨ Features

- *📄 Document Upload*: Support for PDF, TXT, DOCX, and Markdown files
- *🔍 Vector Search*: Intelligent document retrieval using embeddings
- *🌐 Web Search*: Real-time web search integration with SERPAPI
- *🤖 AI-Powered*: Google Gemini AI for natural language responses
- *💬 Dual Response Modes*: Choose between concise or detailed answers
- *📱 Responsive UI*: Modern, professional interface with FontAwesome icons
- *⚡ Real-time Chat*: Interactive chat interface with message history

## 🏗 Project Structure

\\\`
project/
├── config/
│   └── config.py          # API keys and configuration
├── models/
│   ├── llm.py             # Gemini LLM integration
│   └── embeddings.py      # RAG embedding models
├── utils/
│   ├── document_processor.py  # File processing utilities
│   ├── web_search.py         # SERPAPI integration
│   └── rag_pipeline.py       # Complete RAG pipeline
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md             # This file
\\\`

## 🚀 Quick Start

### 1. Clone and Setup

\\\`bash
git clone <your-repo-url>
cd rag-chatbot
pip install -r requirements.txt
\\\`

### 2. Configure API Keys

Edit config/config.py and add your API keys:

\\\`python
# Replace with your actual API keys
GEMINI_API_KEY = "your-gemini-api-key-here"
SERPAPI_KEY = "your-serpapi-key-here"
\\\`

Or set environment variables:
\\\`bash
export GEMINI_API_KEY="your-gemini-api-key-here"
export SERPAPI_KEY="your-serpapi-key-here"
\\\`

### 3. Run the Application

\\\`bash
streamlit run app.py
\\\`

## 🔧 Configuration

### API Keys Required

1. *Google Gemini API*: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. *SERPAPI Key*: Get from [SERPAPI](https://serpapi.com/dashboard)

### Customizable Settings

- *Chunk Size*: Document chunking size (default: 1000)
- *Similarity Threshold*: Vector search threshold (default: 0.7)
- *Max Tokens*: Response length limit (default: 1000)
- *Temperature*: AI creativity level (default: 0.7)

## 📖 Usage

1. *Upload Documents*: Use the sidebar to upload PDF, TXT, DOCX, or MD files
2. *Choose Response Mode*: Select between "Detailed" or "Concise" responses
3. *Ask Questions*: Type questions in the chat interface
4. *Get AI Answers*: Receive responses based on your documents and web search

## 🛠 Technical Details

### RAG Pipeline

1. *Document Processing*: Extract and chunk text from uploaded files
2. *Embedding Creation*: Generate vector embeddings using Sentence Transformers
3. *Vector Search*: Find relevant document chunks using FAISS
4. *Web Search*: Supplement with real-time web results when needed
5. *Response Generation*: Use Gemini AI to create contextual answers

### Key Technologies

- *Streamlit*: Web application framework
- *Google Gemini*: Large language model
- *Sentence Transformers*: Text embeddings
- *FAISS*: Vector similarity search
- *SERPAPI*: Web search integration
- *PyPDF2*: PDF text extraction

## 🎨 UI Features

- *Modern Design*: Professional styling with custom CSS
- *FontAwesome Icons*: Rich iconography throughout the interface
- *Responsive Layout*: Works on desktop and mobile devices
- *Real-time Updates*: Live chat with message history
- *File Management*: Visual file upload and management
- *Statistics Dashboard*: Document and usage statistics

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add environment variables in the Streamlit Cloud dashboard
4. Deploy with one click

### Local Development

\\\`bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-key"
export SERPAPI_KEY="your-key"

# Run the app
streamlit run app.py
\\\`

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for production deployment
- Implement proper error handling for API failures
- Consider rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the GitHub Issues page
- Review the configuration settings
- Ensure all API keys are properly set
- Verify file formats are supported

## 🎯 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced document preprocessing
- [ ] Custom embedding models
- [ ] User authentication
- [ ] Chat history persistence
- [ ] Advanced search filters
- [ ] API endpoint creation
- [ ] Docker containerization

---

Built with ❤ using Streamlit, Gemini AI, and modern web technologies.