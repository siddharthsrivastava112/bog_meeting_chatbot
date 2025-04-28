import { useState } from 'react';
import ChatHeader from './ChatHeader';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';
import axios from 'axios';
export default function ChatBot() {
    const [messages, setMessages] = useState([
        { id: 1, text: 'Hello! How can I help you today?', sender: 'bot' }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    const handleSendMessage =async (messageText) => {
        if (messageText.trim() === '') return;
        // Add user message
        setIsLoading(true);
        const response=await axios.post("http://localhost:8000/ask",{
            query:messageText
        },{ timeout: 300000 });
        if(response)
        {
            setIsLoading(false);
            
            const newBotMessage = {
                id:response.data.answer.length+3,
                text: response.data.answer,
                sender: 'bot'
            };
            setMessages(prev => [...prev, newBotMessage]);
            console.log(response);
        }
        // Simulate bot response
        // setTimeout(() => {
        //     const botResponses = [
        //         "I understand your question about '" + messageText + "'. Let me think...",
        //         "That's an interesting point about '" + messageText + "'. Here's what I know...",
        //         "I've analyzed your query regarding '" + messageText + "'. My response is..."
        //     ];
        //     const randomResponse = botResponses[Math.floor(Math.random() * botResponses.length)];

        //     const newBotMessage = {
        //         id: messages.length + 100,
        //         text: randomResponse,
        //         sender: 'bot'
        //     };
        //     setMessages(prev => [...prev, newBotMessage]);
        //     setIsLoading(false);
        // }, 1500);
    };

    return (
        <div className="flex flex-col h-full bg-gray-50">
            <ChatHeader title="Current Chat" />
            <ChatMessages messages={messages} isLoading={isLoading} />
            <ChatInput onSendMessage={handleSendMessage} />
        </div>
    );
}