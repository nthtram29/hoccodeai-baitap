import OpenAI from 'openai'
import { useEffect, useState } from 'react'

const openAi = new OpenAI({
    baseURL: "http://localhost:1234/v1",
  apiKey: "lm-studio",  
  dangerouslyAllowBrowser: true
})
function isBotMessage(chatMessage) {
  return chatMessage.role === "assistant"
}

function App() {
  const [message, setMessage] = useState('')
  const [chatHistory, setChatHistory] = useState([])

  const submitForm = async (e) =>{
    e.preventDefault();

    setMessage('')

    const userMessage = {role: "user", content: message}
    const waitingBotMessage = {
      role: "assistant",
      content: "Bạn chờ một xíu nhé...",
    };

    const newHistory = [...chatHistory, userMessage, waitingBotMessage];
  setChatHistory(newHistory);
  saveChatHistoryToLocal(newHistory);
    
  try {
    const chatCompletion = await openAi.chat.completions.create({
      messages: [...chatHistory, userMessage], 
      model: "gemma-2-9b-it",
      temperature: 0.7,
      max_tokens: -1,
      stream: false,
    });

    const response = chatCompletion.choices[0].message.content;
    const botMessage = { role: "assistant", content: response };

    const updatedHistory = [...chatHistory, userMessage, botMessage];
    setChatHistory(updatedHistory);
    saveChatHistoryToLocal(updatedHistory); 
  } catch (err) {
    const errorMessage = { role: "assistant", content: `❌ Lỗi API: ${err.message}` };
    const finalHistory = [...updatedHistory.slice(0, -1), errorMessage];
    setChatHistory(finalHistory);
  }
  }

  const saveChatHistoryToLocal = (history) => {
    localStorage.setItem("chat_history", JSON.stringify(history));
  };

  const clearChatHistory = () => {
    setChatHistory([]);
    saveChatHistoryToLocal([]);
  };
  
  useEffect(() => {
    const savedHistory = localStorage.getItem("chat_history");
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory));
    }
  }, []);

  return (
    <div className="bg-gray-100 h-screen flex flex-col">
      <div className="container mx-auto p-4 flex flex-col h-full max-w-2xl">
        <h1 className="text-2xl font-bold mb-4">ChatUI với React + OpenAI</h1>

        <form className="flex" onSubmit={submitForm}>
          <input
            type="text"
            placeholder="Tin nhắn của bạn..."
            className="flex-grow p-2 rounded-l border border-gray-300"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded-r hover:bg-blue-600"
          >
            Gửi tin nhắn
          </button>
        </form>

        <div className="flex-grow overflow-y-auto mt-4 bg-white rounded shadow p-4">
          <div className='text-right'>
            <button
              onClick={clearChatHistory}
              className="m-2 text-right"
            >
              🔄
            </button>
          </div>
          {chatHistory?.map((chatMessage, i) =>(
            <div
              key={i}
              className={`mb-2 ${isBotMessage(chatMessage) ? "text-right" : ""}`}>
                <p className='text-gray-600 text-sm'>{isBotMessage(chatMessage) ? "Bot" : "User"}</p>
                <p className={`p-2 rounded-lg inline-block ${isBotMessage(chatMessage) ? "bg-green-100" : "bg-blue-100"}`}>{chatMessage.content}</p>
              </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
