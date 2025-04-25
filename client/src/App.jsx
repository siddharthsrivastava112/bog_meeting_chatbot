import { useState } from 'react';
// import ChatBot from './components/ChatBot';
// import Sidebar from './components/Sidebar/Sidebar';

function App() {
  const [currentChat, setCurrentChat] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);

  const startNewChat = () => {
    const newChat = {
      id: Date.now(),
      title: `Chat ${chatHistory.length + 1}`,
      date: new Date().toLocaleString()
    };
    setChatHistory([newChat, ...chatHistory]);
    setCurrentChat(newChat);
  };

  return (
    <div className="flex h-screen">
      <Sidebar 
        chatHistory={chatHistory} 
        onNewChat={startNewChat}
        onSelectChat={setCurrentChat}
        currentChatId={currentChat?.id}
      />
      <div className="flex-1 flex flex-col">
        {currentChat ? (
          <ChatBot />
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <button 
              onClick={startNewChat}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg"
            >
              Start New Chat
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
